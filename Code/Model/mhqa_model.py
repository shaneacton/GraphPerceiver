from typing import List

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import BatchEncoding

from Code.Model.bert_embedder import BertEmbedder
from Code.Model.graph_perceiver import GraphPerceiver
from Code.Model.perceiver_mhqa_base import PerceiverMHQA
from Code.Model.scorer import Scorer
from Code.Model.span_embedder import SpanEmbedder
from Code.Transformers.summariser import Summariser
from Code.Utils.model_utils import num_params
from Code.wikipoint import Wikipoint
from Config.options import device, model_conf


class MHQA(PerceiverMHQA):

    def __init__(self):
        super().__init__()
        self.bert = BertEmbedder()
        self.dims = self.bert.dims
        self.summariser = Summariser(self.dims)
        self.span_embedder = SpanEmbedder(self.dims)
        self.perceiver = GraphPerceiver(depth=model_conf().perceiver_layers, dim=self.dims, latent_dim=self.dims,
                                        queries_dim=self.dims, self_per_cross_attn=model_conf().self_per_cross_attn)

        print("params- summariser:", num_params(self.summariser), "perceiver:", num_params(self.perceiver))

        self.candidate_scorer = Scorer(self.dims)
        self.entity_scorer = Scorer(self.dims)

        self.loss_fn = CrossEntropyLoss()
        self.last_epoch = -1
        self.last_i = -1

    def forward(self, wikipoint: Wikipoint):

        """
            first we encode each text into a vector sequence with Bert
            then we summarise each entity node into a fixed size vector
            the node sequence is then fed into the graph perceiver for node correlation with lookups
        """
        nodes, full_embeddings, candidate_summaries, support_encodings = self.get_all_embeddings(wikipoint)
        graph_encoding = self.perceiver(full_embeddings, nodes)  # ~ (e, f)

        return self.finish(graph_encoding, wikipoint, support_encodings=support_encodings)

    def pass_output_model(self, x: Tensor, example: Wikipoint, support_encodings: List[BatchEncoding]):
        num_ents = example.num_ents()
        num_cands = len(example.candidates)
        candidate_ids = list(range(num_ents, num_cands + num_ents))
        cand_embs = x[:, torch.tensor(candidate_ids).to(device).long(), :]
        cand_probs = self.candidate_scorer(cand_embs).view(len(candidate_ids))

        ent_probs = []
        for c, cand in enumerate(example.candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            linked_ent_nodes = set()  # the ent ids this cand is linked to
            for d, token_spans in enumerate(example.ent_token_spans):
                doc_tokens = support_encodings[d].tokens()
                for e, ent_token_span in enumerate(token_spans):
                    """
                        for each combination of candidate and entity, check for a text match
                    """
                    ent_tokens = doc_tokens[ent_token_span[0]:ent_token_span[1]]
                    ent = self.bert.tokenizer.convert_tokens_to_string(ent_tokens)
                    if ent == cand:
                        flat_ent_id = example.get_flat_ent_id(d, e)
                        linked_ent_nodes.add(flat_ent_id)

            if len(linked_ent_nodes) == 0:  # no mentions of this candidate
                ent_prob = torch.tensor(0.).to(device)
            else:  # this cand has some entity mentions in the context
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(device).long()
                ent_embs = torch.index_select(x, dim=1, index=linked_ent_nodes)  # (1, matches, f)
                cand_ent_probs = self.entity_scorer(ent_embs)  # (1, matches, 1)
                ent_prob = torch.max(cand_ent_probs)  # (1)

            ent_probs += [ent_prob]
        ent_probs = torch.stack(ent_probs, dim=0)
        final_probs = cand_probs + ent_probs
        return final_probs