import torch
from torch import Tensor

from Code.Model.mhqa.mhqa_model import MHQAModel
from Code.Model.perceiver.graph_perceiver import GraphPerceiver
from Code.Utils.model_utils import num_params
from Code.wikipoint import Wikipoint
from Config.options import model_conf


class GraphPerceiverMHQA(MHQAModel):

    def __init__(self):
        super().__init__()
        self.perceiver = GraphPerceiver(depth=model_conf().perceiver_layers, dim=self.dims, latent_dim=self.dims,
                                        queries_dim=self.dims, self_per_cross_attn=model_conf().self_per_cross_attn)

        print("params- summariser:", num_params(self.summariser), "perceiver:", num_params(self.perceiver))

    def forward(self, wikipoint: Wikipoint):

        """
            first we encode each text into a vector sequence with Bert
            then we summarise each entity node into a fixed size vector
            the node sequence is then fed into the graph perceiver for node correlation with lookups
        """
        nodes, full_embeddings, candidate_summaries, support_encodings = self.get_all_embeddings(wikipoint)
        candidate_summaries = torch.cat(candidate_summaries).view(1, len(wikipoint.candidates), -1)
        candidate_queries = torch.relu(self.candidate_query_mat(candidate_summaries))
        logits = self.perceiver(full_embeddings, nodes, queries=candidate_queries)  # ~ (e, f)

        return self.finish(logits, wikipoint, support_encodings=support_encodings)

    def pass_output_model(self, latents: Tensor, **kwargs):
        final_probs = self.candidate_scorer(latents)
        return final_probs
