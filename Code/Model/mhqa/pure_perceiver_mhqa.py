import torch
from torch import Tensor

from Code.Model.mhqa.graph_perceiver_mhqa import GraphPerceiverMHQA
from Code.Model.perceiver.perceiver_io import PerceiverIO
from Code.wikipoint import Wikipoint
from Config.options import model_conf


class PerceiverIOMHQA(GraphPerceiverMHQA):

    def __init__(self):
        super().__init__()
        self.perceiver = PerceiverIO(depth=model_conf().perceiver_layers, dim=self.dims, latent_dim=self.dims,
                                        queries_dim=self.dims, self_per_cross_attn=model_conf().self_per_cross_attn)

    def forward(self, wikipoint: Wikipoint):

        """
            first we encode each text into a vector sequence with Bert
            then we summarise each entity node into a fixed size vector
            the node sequence is then fed into the graph perceiver for node correlation with lookups
        """
        _, full_embeddings, candidate_summaries, support_encodings = self.get_all_embeddings(wikipoint)
        candidate_summaries = torch.cat(candidate_summaries).view(1, len(wikipoint.candidates), -1)
        logits = self.perceiver(full_embeddings, queries=candidate_summaries)  # ~ (e, f)

        return self.finish(logits, wikipoint, support_encodings=support_encodings)

    def pass_output_model(self, latents: Tensor, **kwargs):
        final_probs = self.candidate_scorer(latents)
        return final_probs