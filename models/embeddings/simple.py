from typing import Sequence
import torch
import torch.nn as nn
from ..network_utils import build_mlp, build_cnn1d


class MLPObservationEmbeddings(nn.Module):
    """
    Flattens the dict observation (rays + proprio) into a single vector and passes it through an MLP.

    input_dim must equal 7 * num_rays + proprio_dim (4).
    """

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], feature_dim: int):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_sizes, feature_dim)

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # rays:   (B, num_channels, num_rays)
        # proprio:(B, 4)
        x = torch.cat([rays.flatten(1), proprio], dim=-1)
        return self.net(x)


class CNNObservationEmbeddings(nn.Module):
    """
    Processes the dict observation with two separate streams:
      - rays   → 1-D CNN  → feature vector
      - proprio → small MLP → feature vector
    The two vectors are concatenated and projected to feature_dim.

    Args:
        ray_channels:         input channels of the ray matrix (7 for current env)
        cnn_out_channels:     output channels of the 1-D CNN
        proprio_dim:          length of the proprio vector (4)
        proprio_hidden_sizes: hidden sizes for the proprio MLP
        feature_dim:          output dimension after fusion
    """

    def __init__(
        self,
        ray_channels: int,
        cnn_out_channels: int,
        proprio_dim: int,
        proprio_hidden_sizes: Sequence[int],
        feature_dim: int,
    ):
        super().__init__()
        self.cnn        = build_cnn1d(ray_channels, cnn_out_channels)
        self.proprio_net = build_mlp(proprio_dim, proprio_hidden_sizes, proprio_hidden_sizes[-1])
        self.fusion      = nn.Linear(cnn_out_channels + proprio_hidden_sizes[-1], feature_dim)

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # rays:    (B, ray_channels, num_rays)
        # proprio: (B, proprio_dim)
        cnn_feat    = self.cnn(rays).squeeze(-1)      # (B, cnn_out_channels)
        proprio_feat = self.proprio_net(proprio)       # (B, proprio_hidden_sizes[-1])
        return self.fusion(torch.cat([cnn_feat, proprio_feat], dim=-1))  # (B, feature_dim)
