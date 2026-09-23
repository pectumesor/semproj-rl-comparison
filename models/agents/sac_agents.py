import torch
import torch.nn as nn
import torch.optim as optim
from ..heads import GuassianPolicyHead, SquashedGaussianPolicyHead, DoubleQNet, ValueNet
from pathlib import Path
from .base_agent import BaseAgent
from .recurrent_agent import RecurrentAgent

class SACAgent(BaseAgent):
    def __init__(self,
                 obs_embed_model: nn.Module, backbone_model: nn.Module,
                 actor: GuassianPolicyHead | SquashedGaussianPolicyHead,
                 critic: DoubleQNet | ValueNet,
                 action_low, action_high):
        super().__init__(obs_embed_model, backbone_model, actor, critic)

        # tanh squashes to [-1, 1]; an affine map a = center + scale * tanh(u) moves it into the
        # per-dimension env bounds (e.g. forward speed in [0, 1]) instead of clipping half the range away
        action_low  = torch.as_tensor(action_low,  dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale",  (action_high - action_low) / 2.0, persistent=False)
        self.register_buffer("action_center", (action_high + action_low) / 2.0, persistent=False)

    def sample_action(self, obs: torch.Tensor):

        h = self.forward(obs)
        action, action_log_prob = self.actor.act(h)
        # Change of variables for the affine rescale: log p(a') = log p(a) - sum log(scale)
        action_log_prob = action_log_prob - torch.log(self.action_scale).sum()
        return self.action_center + self.action_scale * action, action_log_prob

    def predict_action(self, obs: dict):
        h = self.forward(obs)
        return self.action_center + self.action_scale * self.actor.act_inference(h)

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
