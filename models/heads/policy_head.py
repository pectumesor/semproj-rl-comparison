from ..network_utils import build_mlp
from typing import Sequence
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

LOG_STD_MIN = -5
LOG_STD_MAX = 1.0


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

    def to_env_action(self, action: torch.Tensor) -> torch.Tensor:
        """The sampled action already is the env's continuous (turning, speed) action"""
        return action


class CategoricalPolicyHead(nn.Module):

    """
    Policy Head used for PPO Actor with a discrete action space.

    The policy picks an index into action_table, a fixed list of (turning, speed) pairs, and the
    env still receives the continuous action of that entry (to_env_action). To fit the
    continuous rollout buffers, actions are stored as (B, act_dim) with the index in column 0,
    the mean/std are those of the continuous action under the categorical distribution
    (logging only; the KL of a discrete policy is estimated from the log-prob ratio).
    """

    @property
    def action_mean(self):
        probs = self.distribution.probs                               # (B, N)
        return probs @ self.action_table                              # (B, act_dim)

    @property
    def action_std(self):
        probs = self.distribution.probs
        second_moment = probs @ self.action_table.pow(2)
        return (second_moment - self.action_mean.pow(2)).clamp_min(0.0).sqrt()

    @property
    def entropy(self):
        return self.distribution.entropy()

    def __init__(self, backbone_dim: int, action_table, hidden_sizes: Sequence[int]):
        super().__init__()

        self.register_buffer("action_table", torch.as_tensor(action_table, dtype=torch.float32), persistent=False)
        self.logits = build_mlp(backbone_dim, hidden_sizes, self.action_table.shape[0])
        self.distribution = None

    def forward(self):
        pass

    def update_distribution(self, backbone_feats: torch.Tensor):
        self.distribution = Categorical(logits=self.logits(backbone_feats))

    def _store_index(self, index: torch.Tensor) -> torch.Tensor:
        action = torch.zeros(*index.shape, self.action_table.shape[1], device=index.device)
        action[..., 0] = index.float()
        return action

    def act(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        self.update_distribution(backbone_feats)
        return self._store_index(self.distribution.sample())

    def log_prob_action(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions[..., 0].long())

    def act_inference(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        """Greedy action, already mapped to the env's continuous action"""
        return self.action_table[self.logits(backbone_feats).argmax(dim=-1)]

    def to_env_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_table[action[..., 0].long()]


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



    