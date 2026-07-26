import torch
import torch.nn as nn
from envs.env_utils import *
from typing import Optional, Tuple
from ..heads import GuassianPolicyHead
from ..backbones import SimpleLSTM
import torch.optim as optim
from pathlib import Path

class RecurrentAgent(nn.Module):
    def __init__(self,
                 obs_embed_model: nn.Module, backbone_model: SimpleLSTM,
                 actor: GuassianPolicyHead, critic: nn.Module):
        super().__init__()
        self.obs_embed_model = obs_embed_model
        self.backbone_model = backbone_model
        self.actor = actor
        self.critic = critic

    def forward(self, obs,
                lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):
        
        rays = obs['rays'].reshape(-1, *obs['rays'].shape[-2:])
        proprio = obs['proprio'].reshape(-1, obs['proprio'].shape[-1])
        obs_feat = self.obs_embed_model(rays, proprio)

        hidden, new_lstm_state = self.backbone_model(obs_feat, lstm_state, done)
        return hidden, new_lstm_state
    
    def select_action(self, obs: torch.Tensor, 
                      lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):
        with torch.no_grad():
            hidden, lstm_state = self.forward(obs, lstm_state, done)
            action = self.actor.act(hidden)
            action_log_prob = self.actor.log_prob_action(action)
            action_mu = self.actor.action_mean
            action_std = self.actor.action_std
            value = self.critic(hidden).squeeze(-1)

        return action, action_log_prob, action_mu, action_std, value, lstm_state

    def predict_action(self, obs: torch.Tensor, lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):
        h, new_lstm_state = self.forward(obs, lstm_state, done)
        action = self.actor.act_inference(h)
        return action, new_lstm_state
    
    def get_value(self, obs: torch.Tensor, 
                  lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):
        hidden, _ = self.forward(obs, lstm_state, done)
        return self.critic(hidden).squeeze(-1)
    
    def evaluate_actions(self, obs: torch.Tensor, 
                        lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor,
                        actions: torch.Tensor):
        
        h, _ = self.forward(obs,
                              (lstm_state[0], lstm_state[1]),
                              done)
        
        self.actor.update_distribution(h)
        logp = self.actor.log_prob_action(actions)
        mu = self.actor.action_mean
        std = self.actor.action_std
        entropy = self.actor.entropy
        val = self.critic(h).squeeze(-1)

        return logp, mu, std, entropy, val

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
