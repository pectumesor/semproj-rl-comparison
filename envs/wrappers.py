import gymnasium as gym
import numpy as np
import torch
from .navigation_env import NavigationEnv
from models.embeddings.simple import MLPObservationEmbeddings
from models.backbones.mlp_backbone import MLPBackbone
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavigationEnvSB3(gym.Env):
    def __init__(self, cfg, num_rays, obs_dim, device="cpu"):
        super().__init__()
        self._env = NavigationEnv(cfg, None, num_rays, obs_dim, num_envs=1, device=device)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=obs_dim, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        return obs[0].cpu().numpy(), info

    def step(self, action):
        action_t = torch.tensor(action, dtype=torch.float32, device=self._env.device).unsqueeze(0)
        obs, reward, terminated, truncated, info = self._env.step(action_t)
        return obs[0].cpu().numpy(), reward[0].item(), bool(terminated[0]), bool(truncated[0]), info



class MyBackbone(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int,
                 obs_embed_hidden_sizes, backbone_hidden_sizes):
        super().__init__(observation_space, features_dim)
        obs_flat_dim = int(np.prod(observation_space.shape))
        self.embed = MLPObservationEmbeddings(obs_flat_dim, obs_embed_hidden_sizes,
                                              obs_embed_hidden_sizes[-1])
        self.backbone = MLPBackbone(obs_embed_hidden_sizes[-1], backbone_hidden_sizes,
                                    backbone_hidden_sizes[-1])

    def forward(self, obs):
        return self.backbone(self.embed(obs))
