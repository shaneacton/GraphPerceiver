from typing import List, Tuple

import torch
from torch import nn, Tensor

from Transformers.summariser import Summariser
from bert_embedder import BertEmbedder
from constants import CANDIDATE, DOCUMENT, ENTITY
from graph_perceiver import GraphPerceiver
from model_utils import num_params
from wikipoint import Wikipoint


class MHQA(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertEmbedder()
        self.dims = self.bert.dims
        self.summariser = Summariser(self.dims)
        self.perceiver_io = GraphPerceiver(depth=2, dim=self.dims, queries_dim=self.dims)
        print("params- summariser:", num_params(self.summariser), "perceiver:", num_params(self.perceiver_io))

    def forward(self, wikipoint: Wikipoint):

        """
            first we encode each text into a vector sequence with Bert
            then we summarise each entity node into a fixed size vector
            the node sequence is then fed into the graph perceiver for node correlation with lookups
        """
        support_embeddings, query_emb, cand_embs = self.get_bert_encodings(wikipoint)
        full_embeddings = torch.cat([query_emb] + support_embeddings + cand_embs)

        print("full embs:", full_embeddings.size())

    def get_bert_encodings(self, wikipoint: Wikipoint):
        support_embeddings = [self.bert(sup) for sup in wikipoint.supports]
        query_emb = self.bert(wikipoint.query)
        cand_embs = [self.bert(cand) for cand in wikipoint.candidates]
        return support_embeddings, query_emb, cand_embs

    def get_graph_features(self, wikipoint: Wikipoint, support_embeddings, query_emb, cand_embs):
        """
            performs coattention between the query and context sequence
            then summarises subsequences of tokens according to node spans
            yielding the same-sized node features
        """
        candidate_summaries = self.summariser(cand_embs, CANDIDATE, query_vec=query_emb)
        support_summaries = self.summariser(support_embeddings, DOCUMENT, query_vec=query_emb)

        ent_summaries = self.get_entity_summaries(wikipoint.ent_token_spans, support_embeddings, query_vec=query_emb)
        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        return x

    def get_entity_summaries(self, tok_spans: List[List[Tuple[int]]], support_embeddings: List[Tensor],
                             query_vec=None):
        flat_spans = []
        flat_vecs = []
        for s, spans in enumerate(tok_spans):  # for each support document
            flat_spans.extend(spans)
            flat_vecs.extend([support_embeddings[s]] * len(spans))
        # return [summariser(vec, ENTITY, flat_spans[i]) for i, vec in enumerate(flat_vecs)]
        return self.summariser(flat_vecs, ENTITY, flat_spans, query_vec=query_vec)
