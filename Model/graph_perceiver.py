from torch import nn, Tensor

from einops import repeat
from torch import nn

# helpers
import config
from Model.perceiver_io import exists, PreNorm, FeedForward, cache_fn, Attention


# main class
class GraphPerceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        self_per_cross_attn = 1
    ):
        super().__init__()

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        full_text: Tensor,
        nodes: Tensor,
        mask=None,
    ):
        """
        here nodes are taking the place of our latent space. Number of nodes, and hence the number of latents is variable
        full self attention is performed over the nodes, ie a fully connected graph
        the peek process involves bringing new information into the graph from the full bert token level input

        :param full_text: an (l, dim) tensor
        :param nodes: a (n, dim) tensor
        :param mask:
        :return: the updated node states
        """
        b, *_, device = *full_text.shape, config.device
        x = repeat(nodes, 'n d -> b n d', b=b)

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            # process
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

            # peek
            x = cross_attn(x, context=full_text, mask=mask) + x
            x = cross_ff(x) + x

        return x

