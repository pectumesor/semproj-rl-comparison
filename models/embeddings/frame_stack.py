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
                feature_dim: int):
        super().__init__()

        self.net = build_mlp(input_dim * stack_depth, hidden_sizes, feature_dim)

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
            # Input:
            #   - rays: (stack_size, num_envs, num_channels, num_rays)
            #   - proprio: (stack_size, num_envs, 4)
            _, num_envs, _, _ = rays.shape

            rays_flat = rays.flatten(2) # (F,E,C,R) -> (F,E,C*R)
            x = torch.cat([rays_flat, proprio], dim=-1) # (F,E,C*R+4) == (F,E,input_dim)

            # (F,E,input_dim) -> (E,F,input_dim) -> (E, F*input_dim)
            x = x.permute(1, 0, 2).reshape(num_envs, -1) 
            return self.net(x)
    

class FrameStackCNN(nn.Module):
    """
    Neural Network to create observation embeddings on a stack of current and past frames.

    Treats observations as images and uses CNN's to extract information out of it.
    """

    def __init__(self,
                stack_depth: int,
                out_feature_dim: int):
        super().__init__()

        # TODO: During experiments decide on the appropriate cnn_block depth
        self.block = nn.Sequential(*cnn_block(input_channels= stack_depth, output_channels= stack_depth * 2))
        self.feature_map = nn.LazyLinear(out_feature_dim) # input dim depends on ray_dim, inferred on first forward

    def forward(self, rays: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
            # Input:
            #   - rays: (stack_size, num_envs, num_channels, num_rays)
            #   - proprio: (stack_size, num_envs, 4)
            _, num_envs, _, _ = rays.shape

            rays_img = rays.permute(1, 0, 2, 3) # (F,E,C,R) -> (E,F,C,R): stacked frames become CNN channels
            cnn_feat = self.block(rays_img).flatten(1) # (E, out_channels, H', W') -> (E, out_channels*H'*W')

            proprio_flat = proprio.permute(1, 0, 2).reshape(num_envs, -1) # (F,E,4) -> (E, F*4)

            x = torch.cat([cnn_feat, proprio_flat], dim=-1)
            return self.feature_map(x)
