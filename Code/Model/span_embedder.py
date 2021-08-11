from typing import List, Tuple

import torch
from torch import nn

from Config.options import device


class SpanEmbedder(nn.Module):

    def __init__(self, dims):
        super().__init__()
        span_dim = dims//3
        doc_dim = dims - 2 * span_dim
        self.document_embeddings = nn.Embedding(150, doc_dim)
        self.span_embeddings = nn.Embedding(4098, span_dim)

    def get_embedding_from_ent_spans(self, ent_spans: List[List[Tuple[int]]]):
        span_embs = []
        for d, doc_spans in enumerate(ent_spans):
            starts = [e[0] for e in doc_spans]
            ends = [e[1] for e in doc_spans]
            if len(starts) == 0:  # no ents in this doc
                continue
            span_emb = self(d, starts, ends)
            # print("span emb:", span_emb.size())
            span_embs.append(span_emb)

        end_pos_embs = torch.cat(span_embs, dim=0)
        return end_pos_embs

    def forward(self, doc_id:int, span_starts:List[int], span_ends:List[int]):
        doc_id = torch.Tensor([doc_id]).to(device).long()
        span_starts = torch.Tensor(span_starts).to(device).long()
        span_ends = torch.Tensor(span_ends).to(device).long()

        start_emb = self.span_embeddings(span_starts)
        end_emb = self.span_embeddings(span_ends)
        d_emb = torch.cat([self.document_embeddings(doc_id)] * start_emb.size(0))

        # print("d:", d_emb.size(), "s:", start_emb.size(), "e:", end_emb.size())

        return torch.tanh(torch.cat([d_emb, start_emb, end_emb], dim=1)) * 0.1 #  (num_spans,f)