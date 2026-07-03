import torch
import torch.nn as nn
import torch.optim as optim
from envs.env_utils import *
from ..heads import GuassianPolicyHead, SquashedGaussianPolicyHead, DoubleQNet, ValueNet
from pathlib import Path

class BaseAgent(nn.Module):
    def __init__(self,
                 obs_embed_model: nn.Module, backbone_model: nn.Module,
                 actor: GuassianPolicyHead | SquashedGaussianPolicyHead, 
                 critic: DoubleQNet | ValueNet):
        super().__init__()
        self.obs_embed_model = obs_embed_model
        self.backbone_model = backbone_model
        self.actor = actor
        self.critic = critic


    def forward(self, obs):
        
        obs_feat = self.obs_embed_model(obs['rays'], obs['proprio'])
        h = self.backbone_model(obs_feat)
        return h
    
    def get_value(self, obs: dict):
        return self.critic(self.forward(obs)).squeeze(-1)

    def get_state_action_value(self, obs: dict, actions: torch.Tensor):
        return self.critic(self.forward(obs), actions)
