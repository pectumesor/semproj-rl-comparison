import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..heads import GuassianPolicyHead, DepthHead, LoopClosureHead
from ..backbones import SimpleLSTM
from ..embeddings.frame_stack import FrameStackMLP, FrameStackCNN
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

        obs_feat = self.embed_obs(obs)
        hidden, new_lstm_state = self.backbone_model(obs_feat, lstm_state, done)
        return hidden, new_lstm_state

    def embed_obs(self, obs) -> torch.Tensor:
        # Non-stacked rays/proprio carry only (..., C, R) / (..., proprio_dim) trailing dims.
        # Frame-stacked obs carry an extra stack-size dim: (..., F, C, R) / (..., F, proprio_dim).
        # Either way, flatten every leading dim (time window, envs, ...) into one batch dim
        # while keeping the dims the observation embedding model actually consumes intact.
        frame_stacked = isinstance(self.obs_embed_model, (FrameStackMLP, FrameStackCNN))
        rays_dims = 3 if frame_stacked else 2
        proprio_dims = 2 if frame_stacked else 1

        rays = obs['rays'].reshape(-1, *obs['rays'].shape[-rays_dims:])
        proprio = obs['proprio'].reshape(-1, *obs['proprio'].shape[-proprio_dims:])
        return self.obs_embed_model(rays, proprio)
    
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
        
        hidden, _ = self.forward(obs,
                              (lstm_state[0], lstm_state[1]),
                              done)
        
        self.actor.update_distribution(hidden)
        logp = self.actor.log_prob_action(actions)
        mu = self.actor.action_mean
        std = self.actor.action_std
        entropy = self.actor.entropy
        val = self.critic(hidden).squeeze(-1)

        return logp, mu, std, entropy, val, hidden

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

class RecurrentAgentAuxiliaryHead(RecurrentAgent):

    def __init__(self, depth_head: DepthHead, loop_closure_head: LoopClosureHead, **kwargs):
        super().__init__(**kwargs)

        self.depth_head = depth_head
        self.loop_closure_head = loop_closure_head

    def depth_forward(self, backbone_features: torch.Tensor):
        return self.depth_head(backbone_features)

    def loop_closure_forward(self, backbone_features: torch.Tensor):
        return self.loop_closure_head(backbone_features)

