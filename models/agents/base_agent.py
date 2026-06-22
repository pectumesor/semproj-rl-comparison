import torch
import torch.nn as nn
from envs.env_utils import *

from ..heads import GuassianPolicyHead, SquashedGaussianPolicyHead, QNet, DoubleQNet, ValueNet

class BaseAgent(nn.Module):
    def __init__(self,
                 obs_backbone: nn.Module, policy_backbone: nn.Module,
                 actor: GuassianPolicyHead | SquashedGaussianPolicyHead, 
                 critic: DoubleQNet | ValueNet):
        
        self.obs_backbone = obs_backbone
        self.policy_backbone = policy_backbone
        self.actor = actor
        self.critic = critic


    def forward(self, obs: torch.Tensor):

        obs_feat = self.obs_backbone(obs)
        h = self.policy_backbone(obs_feat)
        return h
    
    def sample_action(self, obs: torch.Tensor):

        with torch.no_grad():
            h = self.forward(obs)
            if isinstance(self.actor, GuassianPolicyHead):
                action = self.actor.act(h)
                action_log_prob = self.actor.log_prob_action(action)
            else:
                action, action_log_prob = self.actor.act(h)
        
        return action, action_log_prob

    
    def select_action(self, obs: torch.Tensor):

        with torch.no_grad():
            h = self.forward(obs)
            action, action_log_prob = self.sample_action(obs)
            action_mu = self.actor.action_mean
            action_std = self.actor.action_std
            value = self.critic(h).squeeze(-1)

        return action, action_log_prob, action_mu, action_std, value

    def predict_action(self, obs: torch.Tensor):
        h = self.forward(obs)
        action = self.actor.act_inference(h)
        return action
    
    def get_value(self, obs: torch.Tensor):

        return self.critic(self.forward(obs))
    
    def get_state_action_value(self, obs: torch.Tensor, actions: torch.Tensor):

        return self.critic(self.forward(obs), actions)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        h = self. forward(obs)
        self.actor.update_distribution(h)
        logp = self.actor.log_prob_action(actions)
        mu = self.actor.action_mean
        std = self.actor.action_std
        entropy = self.actor.entropy
        val = self.critic(h).squeeze(-1)

        return logp, mu, std, entropy, val
    