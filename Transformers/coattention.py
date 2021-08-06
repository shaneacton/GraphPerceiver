from typing import List

import torch
from torch import Tensor
from torch.nn import LayerNorm

from Transformers.transformer import Transformer
from constants import DOCUMENT, QUERY

SOURCE_TYPE_MAP = {DOCUMENT: 0, QUERY: 1}


class Coattention(Transformer):

    """here, the two types are context or query"""

    def __init__(self, intermediate_fac=2, use_type_embedder=True):
        num_types = 2
        super().__init__(conf.embedded_dims, num_types, conf.num_coattention_layers, use_type_embeddings=use_type_embedder,
                         intermediate_fac=intermediate_fac)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, SOURCE_TYPE_MAP)

    def batched_coattention(self, supps: List[Tensor], context_type: str, query: Tensor, return_query_encoding=False) -> List[Tensor]:
        if self.use_type_embeddings:
            supps = [s + self.get_type_tensor(DOCUMENT, s.size(-2)) for s in supps]
            query = (query + self.get_type_tensor(QUERY, query.size(-2)))

        supps = [s.view(-1, self.dims) for s in supps]
        query = query.view(-1, self.dims)

        cats = [torch.cat([supp, query], dim=0) for supp in supps]
        batch, masks = self.pad(cats)
        # print("coat batch:", batch.size(), "num supps:", len(supps))
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        seqs = list(batch.split(dim=0, split_size=1))
        assert len(seqs) == len(supps)
        for s, seq in enumerate(seqs):  # remove padding and query tokens
            last_index = supps[s].size(0) + query.size(0) if return_query_encoding else supps[s].size(0)
            seqs[s] = seq[:, :last_index, :]
            seqs[s] = seqs[s].view(seqs[s].size(1), -1)
        return seqs

    def forward(self, supps: List[Tensor], query: Tensor):
        return self.batched_coattention(supps, query)