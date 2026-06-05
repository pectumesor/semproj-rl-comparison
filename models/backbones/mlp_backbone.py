from typing import Sequence
import torch
import torch.nn as nn
from ..utils import build_mlp

class MLPBackbone(nn.Module):

    """
    Simple MLP Policy Backbone that receives observation embeddings and outputs features for the heads
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Sequence[int],
                 output_dim: int):
        super().__init__()

        self.net = build_mlp(input_dim, hidden_sizes, output_dim)

    def forward(self, observation_feats: torch.Tensor) -> torch.Tensor:
        return self.net(observation_feats)
