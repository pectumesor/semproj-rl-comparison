from typing import Sequence
import torch
import torch.nn as nn
from ..network_utils import build_mlp, build_cnn

class MLPObservationEmbeddings(nn.Module):
    """
    Simple MLP to extract information from agent observations and feed it to the policy backbone
    """

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Sequence[int],
                 feature_dim: int):
        super().__init__()

        self.net = build_mlp(input_dim, hidden_sizes, feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        """
        obs will have dim: (B, obs_dim), where obs_dim is num_classes + 1 x num_rays.

        For a MLP observation embedding we need to flatten it first
        """

        x = obs.flatten(1) # Shape: (B, (num_classes + 1) * num_rays )
        return self.net(x)


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
