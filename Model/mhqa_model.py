from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import BatchEncoding

from Model.bert_embedder import BertEmbedder
from Model.graph_perceiver import GraphPerceiver
from Model.scorer import Scorer
from Transformers.summariser import Summariser
from config import perceiver_layers, device
from constants import CANDIDATE, DOCUMENT, ENTITY
from Utils.model_utils import num_params
from wikipoint import Wikipoint


class MHQA(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertEmbedder()
        self.dims = self.bert.dims
        self.summariser = Summariser(self.dims)
        self.perceiver = GraphPerceiver(depth=perceiver_layers, dim=self.dims, queries_dim=self.dims)
        print("params- summariser:", num_params(self.summariser), "perceiver:", num_params(self.perceiver))

        self.candidate_scorer = Scorer(self.dims)
        self.entity_scorer = Scorer(self.dims)

        self.loss_fn = CrossEntropyLoss()

    def forward(self, wikipoint: Wikipoint):

        """
            first we encode each text into a vector sequence with Bert
            then we summarise each entity node into a fixed size vector
            the node sequence is then fed into the graph perceiver for node correlation with lookups
        """
        support_embeddings, query_emb, cand_embeddings, support_encodings, query_enc, cand_encodings = self.get_bert_encodings(wikipoint)
        # print("query emb:", query_emb.size(), "supps:", [x.size() for x in support_embeddings], "cands:", [x.size() for x in cand_embeddings])
        full_embeddings = torch.cat([query_emb] + support_embeddings + cand_embeddings, dim=1)  # ~ (b, l, f)

        candidate_summaries = [self.summariser(c, CANDIDATE) for c in cand_embeddings]  # ~ (c, f)
        try:
            ents = self.get_entity_summaries(wikipoint.ent_token_spans, support_embeddings)  # ~ (e, f)
        except NoWordsException as e:
            print("wikipoint:", wikipoint)
            print("ent spans:", wikipoint.ent_token_spans)
            raise e
        # print("num ents:", ents.size(0), "num cands:", len(candidate_summaries), "num query tokens:", query_emb.size(1))
        nodes = torch.cat([ents] + candidate_summaries + [query_emb.view(query_emb.size(1), -1)], dim=0)
        graph_encoding = self.perceiver(full_embeddings, nodes)  # ~ (e, f)

        # print("num nodes:", graph_encoding.size(1), "num tokens:", full_embeddings.size(1), " \tprod:", full_embeddings.size(1) * graph_encoding.size(1))

        return self.finish(graph_encoding, wikipoint, support_encodings=support_encodings)

    def finish(self, x: Tensor, example: Wikipoint, **kwargs):
        """performs prediction using the candidate and enity node encodings"""
        final_probs = self.pass_output_model(x, example, **kwargs)

        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)
            return loss, pred_ans

        return pred_ans

    def pass_output_model(self, x: Tensor, example: Wikipoint, support_encodings: List[BatchEncoding]):
        """
            transformations like pooling can change the effective node ids.
            node_id_map maps the original node ids, to the new effective node ids
        """
        num_ents = example.num_ents()
        num_cands = len(example.candidates)
        candidate_ids = list(range(num_ents, num_cands + num_ents))
        cand_embs = x[:, torch.tensor(candidate_ids).to(device).long(), :]
        cand_probs = self.candidate_scorer(cand_embs).view(len(candidate_ids))

        ent_probs = []
        for c, cand in enumerate(example.candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            linked_ent_nodes = set()  # the ent ids this cand is linked to
            for d, token_spans in enumerate(example.ent_token_spans):
                doc_tokens = support_encodings[d].tokens()
                for e, ent_token_span in enumerate(token_spans):
                    """
                        for each combination of candidate and entity, check for a text match
                    """
                    ent_tokens = doc_tokens[ent_token_span[0]:ent_token_span[1]]
                    ent = self.bert.tokenizer.convert_tokens_to_string(ent_tokens)
                    if ent == cand:
                        flat_ent_id = example.get_flat_ent_id(d, e)
                        linked_ent_nodes.add(flat_ent_id)

            if len(linked_ent_nodes) == 0:  # no mentions of this candidate
                ent_prob = torch.tensor(0.).to(device)
            else:  # this cand has some entity mentions in the context
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(device).long()
                ent_embs = torch.index_select(x, dim=1, index=linked_ent_nodes)  # (1, matches, f)
                cand_ent_probs = self.entity_scorer(ent_embs)  # (1, matches, 1)
                ent_prob = torch.max(cand_ent_probs)  # (1)

            ent_probs += [ent_prob]
        ent_probs = torch.stack(ent_probs, dim=0)
        final_probs = cand_probs + ent_probs
        return final_probs

    def get_bert_encodings(self, wikipoint: Wikipoint):
        supports: List[Tensor] = [self.bert(sup) for sup in wikipoint.supports]
        support_embeddings = [s[0] for s in supports]
        support_encodings = [s[1] for s in supports]

        query_emb, query_enc = self.bert(wikipoint.query)
        cands: List[Tensor] = [self.bert(cand) for cand in wikipoint.candidates]
        cand_embeddings = [s[0] for s in cands]
        cand_encodings = [s[1] for s in cands]

        return support_embeddings, query_emb, cand_embeddings, support_encodings, query_enc, cand_encodings

    def get_entity_summaries(self, tok_spans: List[List[Tuple[int]]], support_embeddings: List[Tensor]) -> Tensor:
        flat_spans = []
        flat_vecs = []

        for s, spans in enumerate(tok_spans):  # for each support document
            flat_spans.extend(spans)
            flat_vecs.extend([support_embeddings[s]] * len(spans))

        if len(flat_vecs) == 0:
            raise NoWordsException()
        return self.summariser(flat_vecs, ENTITY, flat_spans)


class NoWordsException(Exception):
    pass