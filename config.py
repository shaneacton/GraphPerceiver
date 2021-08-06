import torch

bert_fine_tune_layers = ["embeddings", "pooler", "LayerNorm"]
bert_size = "small"
device = torch.device("cuda:0")

num_summariser_layers = 1
num_summariser_heads = 8
dropout = 0.2