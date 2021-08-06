from typing import List

import torch
from torch import Tensor
from torch.nn import LayerNorm

from Code.Transformers.switch_transformer import SwitchTransformer
from Code.Transformers.transformer import Transformer
from Code.constants import QUERY, DOCUMENT, ENTITY, CANDIDATE
from Config.config import conf

SOURCE_TYPE_MAP = {DOCUMENT: 0, QUERY: 1}


class SwitchCoattention(SwitchTransformer):

    """here, the two types are context or query"""

    def __init__(self, intermediate_fac=2, use_type_embedder=True):
        super().__init__(conf.embedded_dims, conf.num_coattention_layers, use_type_embeddings=use_type_embedder,
                         intermediate_fac=intermediate_fac,
                         switch_types=[ENTITY, DOCUMENT, CANDIDATE], emb_types=[DOCUMENT, QUERY])

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, SOURCE_TYPE_MAP)

    def batched_coattention(self, context_vectors: List[Tensor], context_type: str, query: Tensor, return_query_encoding=False) -> List[Tensor]:
        """injects query info into various context sequences"""
        if self.use_type_embeddings:  # these type embs differentiate the context and query part of the concattenated sequence
            context_vectors = [s + self.get_type_tensor(DOCUMENT, s.size(-2)) for s in context_vectors]
            query = (query + self.get_type_tensor(QUERY, query.size(-2)))

        context_vectors = [s.view(-1, self.hidden_size) for s in context_vectors]
        query = query.view(-1, self.hidden_size)

        cats = [torch.cat([supp, query], dim=0) for supp in context_vectors]
        batch, masks = Transformer.pad(cats)
        batch = self.switch_encoder(batch, src_key_padding_mask=masks, type=context_type).transpose(0, 1)

        seqs = list(batch.split(dim=0, split_size=1))
        assert len(seqs) == len(context_vectors)
        for s, seq in enumerate(seqs):  # remove padding and query tokens
            last_index = context_vectors[s].size(0) + query.size(0) if return_query_encoding else context_vectors[s].size(0)
            seqs[s] = seq[:, :last_index, :]
            seqs[s] = seqs[s].view(seqs[s].size(1), -1)

        return seqs
