import torch
import torch.nn as nn
import torch.optim as optim
from envs.env_utils import *
from ..heads import GuassianPolicyHead, SquashedGaussianPolicyHead, DoubleQNet, ValueNet
from pathlib import Path
from .base_agent import BaseAgent
from .recurrent_agent import RecurrentAgent

class SACAgent(BaseAgent):
    def __init__(self,
                 obs_embed_model: nn.Module, backbone_model: nn.Module,
                 actor: GuassianPolicyHead | SquashedGaussianPolicyHead,
                 critic: DoubleQNet | ValueNet):
        super().__init__(obs_embed_model, backbone_model, actor, critic)
    
    def sample_action(self, obs: torch.Tensor):

        h = self.forward(obs)
        action, action_log_prob = self.actor.act(h)
        return action, action_log_prob
  
    def predict_action(self, obs: dict):
        h = self.forward(obs)
        return self.actor.act_inference(h)

    def get_value(self, obs: dict):
         raise ValueError("SAC Agent has no Value Net to compute State Values")

    def save_model(self, path, optimizers:dict[str, optim.Optimizer]):

        checkpoint = {
            "obs_embed":self.obs_embed_model.state_dict(),
            "backbone": self.backbone_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor":optimizers["actor"].state_dict(),
            "optimizer_critic":optimizers["critic"].state_dict(),
            "optimizer_alpha":optimizers["alpha"].state_dict()
        }

     
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_model(self, path, device, optimizers: dict[str, optim.Optimizer]):

        checkpoint = torch.load(path, map_location=device)

        self.obs_embed_model.load_state_dict(checkpoint["obs_embed"])
        self.backbone_model.load_state_dict(checkpoint["backbone"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        optimizers["actor"].load_state_dict(checkpoint["optimizer_actor"])
        optimizers["critic"].load_state_dict(checkpoint["optimizer_critic"])
        optimizers["alpha"].load_state_dict(checkpoint["optimizer_alpha"])

class RecurrentSACAgent(RecurrentAgent):
    def __init__(self, obs_embed_model, backbone_model, actor, critic):
        super().__init__(obs_embed_model, backbone_model, actor, critic)
