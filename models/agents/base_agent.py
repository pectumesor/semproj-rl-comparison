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
    
    def save_model(self, path, optimizer: optim.Optimizer):

        checkpoint = {
            "obs_embed":self.obs_embed_model.state_dict(),
            "backbone": self.backbone_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": optimizer.state_dict()
        }

     
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_model(self, path, device, optimizer: optim.Optimizer):

        checkpoint = torch.load(path, map_location=device)

        self.obs_embed_model.load_state_dict(checkpoint["obs_embed"])
        self.backbone_model.load_state_dict(checkpoint["backbone"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])



