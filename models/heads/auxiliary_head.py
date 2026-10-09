from ..network_utils import build_mlp
from typing import Sequence
import torch
import torch.nn as nn


class DepthHead(nn.Module):
    """
    Head for the auxiliary tasks of predicting depth of the ray cast

    Arg:
        feature_dim: Dimension of the input features: (B, feature_dim)
        hidden_sizes: Sizes of the intermediate features
        Output_dim: Dimension of the depth prediction map: (B, num_rays)

    Output is passed through a sigmoid, bounding it to [0, 1] like the normalized ray distances.
    """

    def __init__(self, feature_dim: int, hidden_sizes: Sequence[int], output_dim: int):
        super().__init__()

        self.model = build_mlp(feature_dim, hidden_sizes, output_dim)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(backbone_features))

class LoopClosureHead(nn.Module):
    """
    Head for the auxiliary task of loop closure. Outputs a single logit: (B, 1).
    Sigmoid is not applied here, it is applied inside the loss (binary_cross_entropy_with_logits);
    apply torch.sigmoid on the output only to get the loop closure probability.

    Args:
        feature_dim: Dimension of the input features: (B, feature_dim)
        hidden_sizes: Sizes of the intermediate features
    """
    def __init__(self, feature_dim: int, hidden_sizes: int):
        super().__init__()

        self.net = build_mlp(feature_dim, hidden_sizes, 1)

    def forward(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        return self.net(backbone_feats)
