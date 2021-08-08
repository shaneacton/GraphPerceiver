import torch

"""MODEL options"""
perceiver_layers = 5


"""EMBEDDING options"""
# embedder_name = "google/bigbird-roberta-base"  # "prajjwal1/bert-" for configurable sized bert
embedder_name = "prajjwal1/bert-"  # "prajjwal1/bert-" for configurable sized bert

bert_fine_tune_layers = ["embeddings", "pooler", "LayerNorm", ""]  # "" for all
bert_size = "medium"  # tiny | mini | small | medium


"""GRAPH ENCODING options"""
num_summariser_layers = 1
num_summariser_heads = 8
use_detected_ents = False
use_special_ents = True

"""TRAINING options"""
lr = 0.001
weight_decay = 0.00001
print_loss_every = 100
max_examples = 2001  # -1 for off
device = torch.device("cuda:0")
dropout = 0.1

