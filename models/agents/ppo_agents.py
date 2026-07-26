import torch
import torch.nn as nn
import torch.optim as optim
from envs.env_utils import *
from ..heads import GuassianPolicyHead, SquashedGaussianPolicyHead, DoubleQNet, ValueNet
from pathlib import Path
from .base_agent import BaseAgent
from .recurrent_agent import RecurrentAgent
from typing import Optional, Tuple


"""
    Interface agent to learn via PPO Algorithm

"""

class PPOAgent(BaseAgent):
    def __init__(self,
                 obs_embed_model: nn.Module, backbone_model: nn.Module,
                 actor: GuassianPolicyHead, critic: ValueNet,
                 action_low: int, action_high: int):
        super().__init__(obs_embed_model, backbone_model, actor, critic)

        self.action_low = action_low
        self.action_high = action_high

    def sample_action(self, obs: torch.Tensor):

        h = self.forward(obs)
        action = self.actor.act(h)
        action_log_prob = self.actor.log_prob_action(action)
    
        return action, action_log_prob
  
    def select_action(self, obs: dict):

        with torch.no_grad():
            h = self.forward(obs)
            action = self.actor.act(h)
            action_clipped = action.clamp(min=self.action_low, max=self.action_high)
            action_log_prob = self.actor.log_prob_action(action)
            value = self.critic(h).squeeze(-1)
            action_mu  = self.actor.action_mean
            action_std = self.actor.action_std
          
        return action, action_clipped, action_log_prob, action_mu, action_std, value

    def predict_action(self, obs: dict):
        h = self.forward(obs)
        action = self.actor.act_inference(h)
        return action.clamp(self.action_low, self.action_high)

    def get_state_action_value(self, obs: dict, actions: torch.Tensor):
        raise ValueError("PPO Agent has no QNet or Double QNet to compute State-Action Values")

class RecurrentPPOAgent(RecurrentAgent):

    def __init__(self, obs_embed_model,
                  backbone_model, actor, critic, 
                  action_low, action_high):
        
        super().__init__(obs_embed_model, backbone_model, actor, critic)

        self.action_low = action_low
        self.action_high = action_high

    def select_action(self, obs: torch.Tensor, 
                      lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):
        with torch.no_grad():
            action, action_log_prob, action_mu, action_std, value, lstm_state = super().select_action(obs, lstm_state, done)
            action_clipped = action.clamp(min=self.action_low, max=self.action_high)
       
        return action, action_clipped, action_log_prob, action_mu, action_std, value, lstm_state

