from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from Code.Model.bert_embedder import BertEmbedder
from Code.Model.graph_perceiver import GraphPerceiver
from Code.Model.scorer import Scorer
from Code.Model.span_embedder import SpanEmbedder
from Code.Transformers.summariser import Summariser
from Config.options import device, model_conf
from Code.constants import CANDIDATE, ENTITY
from Code.Utils.model_utils import num_params
from Code.wikipoint import Wikipoint


class MHQA2(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertEmbedder()
        self.dims = self.bert.dims
        self.summariser = Summariser(self.dims)
        self.span_embedder = SpanEmbedder(self.dims)
        self.perceiver = GraphPerceiver(depth=model_conf().perceiver_layers, dim=self.dims, latent_dim=self.dims,
                                        queries_dim=self.dims, self_per_cross_attn=model_conf().self_per_cross_attn)

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
        candidate_summaries = [self.summariser(c, CANDIDATE) for c in cand_embeddings]  # ~ (c, f)
        try:
            ents = self.get_entity_summaries(wikipoint.ent_token_spans, support_embeddings)  # ~ (e, f)
        except NoWordsException as e:
            print("wikipoint:", wikipoint)
            print("ent spans:", wikipoint.ent_token_spans)
            raise e
        nodes = torch.cat([ents] + candidate_summaries + [query_emb.view(query_emb.size(1), -1)], dim=0)
        full_embeddings = torch.cat([query_emb] + support_embeddings + cand_embeddings, dim=1)  # ~ (b, l, f)

        candidate_summaries = torch.cat(candidate_summaries).view(1, len(wikipoint.candidates), -1)
        logits = self.perceiver(full_embeddings, nodes, queries=candidate_summaries)  # ~ (e, f)

        return self.finish(logits, wikipoint, support_encodings=support_encodings)

    def finish(self, logits: Tensor, example: Wikipoint, **kwargs):
        """performs prediction using the candidate and enity node encodings"""
        final_probs = self.candidate_scorer(logits)

        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)
            return loss, pred_ans

        return pred_ans

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