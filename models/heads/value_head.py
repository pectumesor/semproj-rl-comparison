from ..utils import build_mlp
from typing import Sequence
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, backbone_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.qnet = build_mlp(backbone_dim + action_dim, hidden_sizes, 1)

    def forward(self, backbone_feats: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([backbone_feats, actions], dim=1)
        return self.qnet(x)
    
class DoubleQNet(nn.Module):
    def __init__(self, backbone_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.qnet_online = build_mlp(backbone_dim + action_dim, hidden_sizes, 1)
        self.qnet_target = build_mlp(backbone_dim + action_dim, hidden_sizes, 1)
    
    def forward(self, backbone_feats: torch.Tensor, actions: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([backbone_feats, actions], dim=1)
        qval_online = self.qnet_online(x)
        qval_target = self.qnet_target(x)

        return qval_online, qval_target

class ValueNet(nn.Module):
    def __init__(self, backbone_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = build_mlp(backbone_dim,hidden_sizes,1)

    def forward(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        return self.net(backbone_feats)
