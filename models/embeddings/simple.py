from typing import Sequence
import torch
import torch.nn as nn
from ..utils import build_mlp, build_cnn

class MLPObservationEmbeddings(nn.Module):
    """
    Simple MLP to extract information from agent observations and feed it to the policy backbone
    """

    def __init__(self,
                 obs_dim: int,
                 hidden_sizes: Sequence[int],
                 feature_dim: int):
        super().__init__()

        self.net = build_mlp(obs_dim, hidden_sizes, feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CNNObservationEmbeddings(nn.Module):

    """
    Simple MLP Vision Module to extract information from agent observations and feed it to the policy backbone
    """

    def __init__(self,
                input_channels,
                output_channels
                ):
        super().__init__()

        self.net = build_cnn(input_channels, output_channels)

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
