import time
from abc import abstractmethod
from torch import nn


class StringEmbedder(nn.Module):

    @abstractmethod
    def embed(self, string, **kwargs):
        pass

    def forward(self, string):
        emb = self.embed(string)
        return emb