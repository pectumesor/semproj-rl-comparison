from ..utils import build_mlp
from typing import Sequence
import torch
import torch.nn as nn


class DepthHead(nn.Module):
    """
    Head for the auxiliary tasks of predicting depth of images

    Arg:
        feature_dim: Dimension of the input features: (B, feature_dim)
        hidden_sizes: Sizes of the intermediate features
        Output_dim: Dimension of the depth prediction map: (B, H, W)
    """

    def __init__(self, feature_dim: int, hidden_sizes: Sequence[int], output_dim: Sequence[int]):
        super().__init__()

        H,W = output_dim
        self.backbone = build_mlp(feature_dim, hidden_sizes, H)
        self.depth_map = nn.Conv2d(in_channels=1, out_channels=W, kernel_size=1)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:

        h = self.backbone(backbone_features)
        h = h.unsqueeze(-1)
        return self.depth_map(h)
    
class LoopClosureHead(nn.Module):
    """
    Head for the auxiliary task of loop closure. Output dimesion is fixed at 2.

    Args:
        feature_dim: Dimension of the input features: (B, feature_dim)
        hidden_sizes: Sizes of the intermediate features
    """
    def __init__(self, feature_dim: int, hidden_sizes: int):
        super().__init__()

        self.net = build_mlp(feature_dim, hidden_sizes, 2)

    def forward(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        return self.net(backbone_feats)
