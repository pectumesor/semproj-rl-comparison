import gymnasium as gym
import numpy as np
import torch
from .navigation_env import NavigationEnvEasy
from models.embeddings.simple import MLPObservationEmbeddings
from models.backbones.mlp_backbone import MLPBackbone
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavigationEnvSB3(gym.Env):
    """
    Single-env wrapper around NavigationEnv for Stable Baselines 3.

    The dict observation {"rays": (C, R), "proprio": (4,)} is flattened into a
    single numpy vector of shape (C*R + 4,) so SB3 sees a standard Box space.
    """

    def __init__(self, cfg, num_rays: int, ray_dim: tuple, proprio_dim: int, device: str = "cpu"):
        super().__init__()
        self._env       = NavigationEnvEasy(cfg, None, num_rays, ray_dim, num_envs=1, device=device)
        self._ray_dim   = ray_dim
        self._proprio_dim = proprio_dim
        flat_dim = int(np.prod(ray_dim)) + proprio_dim

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(flat_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        rays   = obs["rays"][0].cpu().numpy().flatten()   # (C*R,)
        proprio = obs["proprio"][0].cpu().numpy()          # (4,)
        return np.concatenate([rays, proprio]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        return self._flatten_obs(obs), info

    def step(self, action):
        action_t = torch.tensor(action, dtype=torch.float32,
                                device=self._env.device).unsqueeze(0)
        obs, reward, terminated, truncated, info = self._env.step(action_t)
        return self._flatten_obs(obs), reward[0].item(), bool(terminated[0]), bool(truncated[0]), info


class MyBackbone(BaseFeaturesExtractor):
    """
    SB3 features extractor that mirrors the custom PPO architecture:
      flat obs → split into rays + proprio → MLPObservationEmbeddings → MLPBackbone
    """

    def __init__(self, observation_space: gym.Space, features_dim: int,
                 ray_dim: tuple, proprio_dim: int,
                 obs_embed_hidden_sizes, backbone_hidden_sizes):
        super().__init__(observation_space, features_dim)
        self._ray_dim    = ray_dim
        self._proprio_dim = proprio_dim
        flat_ray_dim = int(np.prod(ray_dim))

        self.embed    = MLPObservationEmbeddings(
            flat_ray_dim + proprio_dim, obs_embed_hidden_sizes, obs_embed_hidden_sizes[-1])
        self.backbone = MLPBackbone(
            obs_embed_hidden_sizes[-1], backbone_hidden_sizes, backbone_hidden_sizes[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C*R + proprio_dim) — flat vector from SB3
        flat_ray_dim = int(np.prod(self._ray_dim))
        rays_flat = obs[:, :flat_ray_dim]
        proprio   = obs[:, flat_ray_dim:]
        rays = rays_flat.view(obs.shape[0], *self._ray_dim)
        return self.backbone(self.embed(rays, proprio))
