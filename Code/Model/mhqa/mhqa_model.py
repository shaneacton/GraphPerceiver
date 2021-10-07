from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from Code.Model.bert_embedder import BertEmbedder
from Code.Model.scorer import Scorer
from Code.Transformers.summariser import Summariser
from Code.Utils.model_utils import num_params
from Code.constants import ENTITY, CANDIDATE
from Code.wikipoint import Wikipoint
from Config.options import device


class MHQAModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertEmbedder()
        self.dims = self.bert.dims
        self.summariser = Summariser(self.dims)
        self.candidate_scorer = Scorer(self.dims)

        self.loss_fn = CrossEntropyLoss()
        self.last_epoch = -1
        self.last_i = -1

    def get_all_embeddings(self, wikipoint: Wikipoint):
        support_embeddings, query_emb, cand_embeddings, support_encodings, query_enc, cand_encodings = self.get_bert_encodings(
            wikipoint)
        candidate_summaries = [self.summariser(c, CANDIDATE) for c in cand_embeddings]  # ~ (c, f)
        try:
            ents = self.get_entity_summaries(wikipoint.ent_token_spans, support_embeddings)  # ~ (e, f)
        except NoWordsException as e:
            print("wikipoint:", wikipoint)
            print("ent spans:", wikipoint.ent_token_spans)
            raise e
        nodes = torch.cat([ents] + candidate_summaries + [query_emb.view(query_emb.size(1), -1)], dim=0)
        full_embeddings = torch.cat([query_emb] + support_embeddings + cand_embeddings, dim=1)  # ~ (b, l, f)
        return nodes, full_embeddings, candidate_summaries, support_encodings

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

    def finish(self, latents: Tensor, example: Wikipoint, **kwargs):
        """performs prediction using the candidate and enity node encodings"""
        final_probs = self.pass_output_model(latents, example=example, **kwargs)
        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)
            return loss, pred_ans

        return pred_ans


class NoWordsException(Exception):
    pass