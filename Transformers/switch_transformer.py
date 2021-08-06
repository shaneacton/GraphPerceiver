from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Code.Embedding.positional_embedder import PositionalEmbedder
from Code.HDE.switch_module import SwitchModule
from Code.Transformers.transformer import Transformer
from Config.config import conf


class SwitchTransformer(nn.Module):

    def __init__(self, hidden_size, num_layers, types=None, intermediate_fac=2, include_global=False,
                 use_type_embeddings=True, use_pos_embeddings=False, switch_types=None, emb_types=None):
        super().__init__()
        self.num_heads = conf.heads
        if types is not None:
            switch_types = types
            emb_types = types

        self.hidden_size = hidden_size
        self.include_global = include_global
        self.use_pos_embeddings = use_pos_embeddings

        encoder_layer = TransformerEncoderLayer(self.hidden_size, conf.transformer_heads,
                                                self.hidden_size * intermediate_fac, conf.dropout, 'relu')
        encoder_norm = LayerNorm(self.hidden_size)
        encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.switch_encoder = SwitchModule(encoder, types=switch_types, include_global=include_global)

        use_type_embeddings = use_type_embeddings and include_global
        self.use_type_embeddings = use_type_embeddings
        if use_type_embeddings:
            self.type_embedder = nn.Embedding(len(emb_types) - 1, hidden_size)
            self.type_emb_norm = LayerNorm(hidden_size)

        if use_pos_embeddings:
            self.pos_embedder = PositionalEmbedder(hidden_size)

    def get_type_tensor(self, type, length, type_map):
        ids = Transformer.get_type_ids(type, length, type_map)
        return self.type_embedder(ids).view(1, -1, self.hidden_size)