import torch
import torch.nn as nn
from envs.env_utils import *
from typing import Optional, Tuple
from ..heads import GuassianPolicyHead

class RecurrentAgent(nn.Module):
    def __init__(self,
                 obs_backbone: nn.Module, policy_backbone: nn.Module,
                 actor: GuassianPolicyHead, critic: nn.Module):
        
        self.obs_backbone = obs_backbone
        self.policy_backbone = policy_backbone
        self.actor = actor
        self.critic = critic

    def forward(self, obs: torch.Tensor,
                lstm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):

        obs_feat = self.obs_backbone(obs)
        hidden, new_lstm_state = self.policy_backbone(obs_feat, lstm_state, done)
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
    