import torch
from torch import nn
from utils.registry import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class TestNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)
