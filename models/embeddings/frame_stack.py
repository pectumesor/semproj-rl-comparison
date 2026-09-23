from typing import Sequence
import torch
import torch.nn as nn
from ..network_utils import build_mlp, cnn_block

class FrameStackMLP(nn.Module):
    """
    Neural Network to create observation embeddings on a stack of current and past frames
    """

    def __init__(self,
                input_dim: int,
                stack_depth: int,
                hidden_sizes: Sequence[int],
                out_feature_dim: int):
        super().__init__()

        self.net = build_mlp(input_dim * stack_depth, hidden_sizes, out_feature_dim)

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
            # Input:
            #   - rays: (num_envs, stack_size, num_channels, num_rays)
            #   - proprio: (num_envs, stack_size, 4)
            num_envs, _, _, _ = rays.shape

            rays_flat = rays.flatten(2) # (E,F,C,R) -> (E,F,C*R)
            x = torch.cat([rays_flat, proprio], dim=-1) # (E,F,C*R+4) == (E,F,input_dim)
            x = x.reshape(num_envs, -1) # (E,F,input_dim) -> (E, F*input_dim)
            return self.net(x)
    

class FrameStackCNN(nn.Module):
    """
    Neural Network to create observation embeddings on a stack of current and past frames.

    Treats observations as images and uses CNN's to extract information out of it.
    """

    def __init__(self,
                stack_depth: int,
                ray_channels: int,
                cnn_out_channels: int,
                out_feature_dim: int):
        super().__init__()

        # TODO: During experiments decide on the appropriate cnn_block depth
        self.block = nn.Sequential(*cnn_block(input_channels= stack_depth * ray_channels, output_channels= cnn_out_channels))
        self.feature_map = nn.LazyLinear(out_feature_dim) # input dim depends on ray_dim, inferred on first forward

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
            # Input:
            #   - rays: (num_envs, stack_size, num_channels, num_rays)
            #   - proprio: (num_envs, stack_size, 4)
            num_envs = rays.shape[0]

            # Stacked frames x ray encodings become the conv channels; convolve along the ray axis only
            rays = rays.flatten(1, 2) # (E,F,C,R) -> (E, F*C, R)
            cnn_feat = self.block(rays).flatten(1) # (E, out_channels, R/2) -> (E, out_channels*R/2)
            proprio_flat = proprio.reshape(num_envs, -1) # (E,F,4) -> (E, F*4)

            x = torch.cat([cnn_feat, proprio_flat], dim=-1)
            return self.feature_map(x)
