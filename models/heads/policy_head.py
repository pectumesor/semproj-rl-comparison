from ..network_utils import build_mlp
from typing import Sequence
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GuassianPolicyHead(nn.Module):

    """
    Policy Head used for PPO Actor
    """
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def __init__(self, backbone_dim: int, actions_dim:int, hidden_sizes: Sequence[int]):
        super().__init__()

        self.mu = build_mlp(backbone_dim, hidden_sizes, actions_dim)
        self.log_std = nn.Parameter(torch.zeros(actions_dim))
        self.distribution = None

    def forward(self):
        pass

    def update_distribution(self, backbone_feats: torch.Tensor):

        """
        Updates the classes self.distribution feature to N(\mu(feats), \Sigma_\phi)
        """

        mean = self.mu(backbone_feats)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)).expand_as(mean)
        self.distribution = Normal(mean, std)

    
    def act(self, backbone_feats: torch.Tensor) -> torch.Tensor:

        """
            Sample actions from the current self.distribution \pi_\theta(a|s)
        """
        self.update_distribution(backbone_feats)
        return self.distribution.sample()
    
    def log_prob_action(self, actions: torch.Tensor):

        """
        Compute the log probability of action 
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, backbone_feats: torch.Tensor) -> torch.Tensor:

        """
            Test the policy by taking the mean action: a ~ \pi(. | s)  
        """
        return self.mu(backbone_feats)


class SquashedGaussianPolicyHead(nn.Module):

    """
    Policy Head used for SAC Actor
    """

    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def __init__(self, backbone_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        
        self.policy_backbone = build_mlp(backbone_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], action_dim)
        self.distribution = None

    def forward(self):
        pass

    def update_distribution(self, backbone_feats: torch.Tensor):

        """
        Update the distribution \pi_\theta(\mu(s), \Sigma_\phi)
        """

        h = self.policy_backbone(backbone_feats)
        mean = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        self.distribution = Normal(mean, std)

    def act(self, bacbkone_feats: torch.Tensor) -> torch.Tensor:

        """
        Sample action from distibution a = tanh(u), u ~ \pi(.|s) and squashed log probabilites
        """

        self.update_distribution(bacbkone_feats)
        u =  self.distribution.rsample()
        log_u = self.distribution.log_prob(u)
        a = torch.tanh(u)
        log_a = log_u - torch.log(1 - a.pow(2) + 1e-6)

        return a, log_a.sum(dim=-1)


    def act_inference(self, backbone_feats: torch.Tensor) -> torch.Tensor:
       h = self.policy_backbone(backbone_feats)
       u =  self.mu(h) # u ~ \pi(.|s)
       return torch.tanh(u) # a = tanh(u)



    