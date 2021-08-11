from typing import List, Union

from torch import Tensor
from transformers import TokenSpan

from Config.options import model_conf
from Code.Transformers.transformer import Transformer
from Code.constants import ENTITY, DOCUMENT, CANDIDATE

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class Summariser(Transformer):
    """
        a summarising transformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding.

        here the 3 types are the node types {entity, document, candidate}
    """

    def __init__(self, dims, intermediate_fac=2):
        num_types = 3
        super().__init__(dims, model_conf().num_summariser_heads, num_types,
                         model_conf().num_summariser_layers, intermediate_fac=intermediate_fac)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    @staticmethod
    def get_vec_extract(full_vec, span):
        if span is None:
            span = (0, full_vec.size(-2))  # full span
        vec = full_vec[:, span[0]: span[1], :].clone()
        return vec

    def forward(self, vec_or_vecs: Union[List[Tensor], Tensor], _type, spans: List[TokenSpan]=None) -> Tensor:
        """
            either one vec shaped (b, seq, f)
            or a vecs list containing (1, seq, f)
            summaries are returned as (b, f)

            if spans is not None, it is a list of token index tuples (start,end), one for each vec
            only these subsequences will be summarised

            if spans is none, the full sequences are summarised
        """
        vecs = vec_or_vecs
        if isinstance(vec_or_vecs, Tensor):
            """break it up into a list of vecs (1, seq, f)"""
            vecs = vec_or_vecs.split(1, dim=0)
            # print("splitting vecs:", vec_or_vecs.size(), [v.size() for v in vecs])
        else:
            if len(list(vecs[0].size())) != 3:  # no batch dim
                vecs = [v.view(1, v.size(0), -1) for v in vecs]
            # print("list vecs:", [v.size() for v in vecs])

        if spans is None:
            spans = [None] * len(vecs)

        extracts = [self.get_vec_extract(v, spans[i]).view(-1, self.dims) for i, v in enumerate(vecs)]
        extracts = [ex + self.get_type_tensor(_type, ex.size(-2)).view(-1, self.dims) for ex in extracts]

        batch, masks = self.pad(extracts)
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        # print("batch:", batch.size())
        summaries = batch[:, 0, :]  # (nodes, hidd)
        # print("summs:", summaries.size())
        return summaries