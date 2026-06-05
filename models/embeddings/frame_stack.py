from typing import Sequence
import torch
import torch.nn as nn
from ..utils import build_mlp, cnn_block

class FrameStackMLP(nn.Module):
    """
    Neural Network to create observation embeddings on a stack of current and past frames
    """

    def __init__(self,
                obs_dim: int,
                stack_depth: int,
                hidden_sizes: Sequence[int],
                feature_dim: int):
        super().__init__()

        self.net = build_mlp(obs_dim * stack_depth, hidden_sizes, feature_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
    

class FrameStackCNN(nn.Module):
    """
    Neural Network to create observation embeddings on a stack of current and past frames.

    Treats observations as images and uses CNN's to extract information out of it.
    """

    def __init__(self,
                stack_depth: int,
                out_feature_dim: int):
        super().__init__()

        # TODO: During experiments decide on the appropriate cnn_block depth and find final flattened feature dimension
        self.block = cnn_block(input_channels= stack_depth, output_channels= stack_depth * 2)
        self.feature_map = nn.Linear(128, out_feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        cnn_feat = self.block(obs)
        cnn_feat = cnn_feat.flatten()

        return self.feature_map(cnn_feat)
 

