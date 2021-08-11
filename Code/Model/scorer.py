import torch
from torch import nn


class Scorer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)

    def forward(self, vec):
        vec = torch.tanh(self.linear1(vec))
        return torch.sigmoid(self.linear2(vec))