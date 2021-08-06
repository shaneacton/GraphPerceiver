import time
from typing import List, Union

import torch
from torch import Tensor
from transformers import TokenSpan

from Transformers.transformer import Transformer
from constants import ENTITY, DOCUMENT, CANDIDATE

from config import num_summariser_layers, num_summariser_heads

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class Summariser(Transformer):
    """
        a summarising transformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding.

        here the 3 types are the node types {entity, document, candidate}
    """

    def __init__(self, dims, intermediate_fac=2):
        num_types = 3
        super().__init__(dims, num_summariser_heads, num_types, num_summariser_layers,
                         use_type_embeddings=False, intermediate_fac=intermediate_fac)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    @staticmethod
    def get_vec_extract(full_vec, span):
        if span is None:
            span = (0, full_vec.size(-2))  # full span
        vec = full_vec[:, span[0]: span[1], :].clone()
        return vec

    def forward(self, vec_or_vecs: Union[List[Tensor], Tensor], _type, spans: List[TokenSpan]=None,
                return_list=True, query_vec: Tensor = None):
        """
            either one vec shaped (b, seq, f)
            or a vecs list containing (1, seq, f)
            summaries are returned as a (1, f) list or (b, f)

            if spans is not None, it is a list of token index tuples (s,e), one for each vec
            only these subsequences will be summarised

            if spans is none, the full sequences are summarised
        """
        vecs = vec_or_vecs
        if isinstance(vec_or_vecs, Tensor):
            """break it up into a list of vecs (1, seq, f)"""
            vecs = vec_or_vecs.split(1, dim=0)

        if spans is None:
            spans = [None] * len(vecs)

        extracts = [self.get_vec_extract(v, spans[i]).view(-1, self.dims) for i, v in enumerate(vecs)]

        batch, masks = self.pad(extracts)
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        summaries = batch[:, 0, :]  # (ents, hidd)

        if return_list:
            return list(summaries.split(dim=0, split_size=1))

        return summaries