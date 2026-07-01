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
    
    def sample_action(self, obs: torch.Tensor):

        h = self.forward(obs)
        if isinstance(self.actor, GuassianPolicyHead):
            action = self.actor.act(h)
            action_log_prob = self.actor.log_prob_action(action)
        else:
            action, action_log_prob = self.actor.act(h)
        
        return action, action_log_prob
  
    def select_action(self, obs: dict):

        with torch.no_grad():
            h = self.forward(obs)
            if isinstance(self.actor, GuassianPolicyHead):
                action = self.actor.act(h)
                action_log_prob = self.actor.log_prob_action(action)
                value = self.critic(h).squeeze(-1)
            else:
                action, action_log_prob = self.actor.act(h)
                value = self.critic(h, action).squeeze(-1)
            action_mu  = self.actor.action_mean
            action_std = self.actor.action_std
          
        return action, action_log_prob, action_mu, action_std, value

    def predict_action(self, obs: dict):
        h = self.forward(obs)
        return self.actor.act_inference(h)

    def get_value(self, obs: dict):
        return self.critic(self.forward(obs)).squeeze(-1)

    def get_state_action_value(self, obs: dict, actions: torch.Tensor):
        return self.critic(self.forward(obs), actions)


    def save_model(self, path, optimizer: optim.Optimizer | dict[str, optim.Optimizer]):

        if isinstance(optimizer, optim.Optimizer):
            self.save_model_ppo(path, optimizer)
        else:
            self.save_model_sac(path, optimizer)
    
    def load_model(self, path, device, optimizer: optim.Optimizer | dict[str, optim.Optimizer]):

        if isinstance(optimizer, optim.Optimizer):
            self.load_model_ppo(path, device, optimizer)
        else:
            self.load_model_sac(path, device, optimizer)

    def save_model_sac(self, path, optimizers:dict[str, optim.Optimizer]):

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

    def save_model_ppo(self, path, optimizer: optim.Optimizer):

        checkpoint = {
            "obs_embed":self.obs_embed_model.state_dict(),
            "backbone": self.backbone_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": optimizer.state_dict()
        }

     
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_model_ppo(self, path, device, optimizer: optim.Optimizer):

        checkpoint = torch.load(path, map_location=device)

        self.obs_embed_model.load_state_dict(checkpoint["obs_embed"])
        self.backbone_model.load_state_dict(checkpoint["backbone"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    def load_model_sac(self, path, device, optimizers: dict[str, optim.Optimizer]):

        checkpoint = torch.load(path, map_location=device)

        self.obs_embed_model.load_state_dict(checkpoint["obs_embed"])
        self.backbone_model.load_state_dict(checkpoint["backbone"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        optimizers["actor"].load_state_dict(checkpoint["optimizer_actor"])
        optimizers["critic"].load_state_dict(checkpoint["optimizer_critic"])
        optimizers["alpha"].load_state_dict(checkpoint["optimizer_alpha"])
