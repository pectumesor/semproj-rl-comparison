import gymnasium as gym
import numpy as np
import torch
import pygame
from .navigation_env import NavigationEnvEasy
from models.embeddings.simple import MLPObservationEmbeddings
from models.backbones.mlp_backbone import MLPBackbone
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .env_utils import w2s

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
    
    def render(self, obs, title, mode="human"):

        _SCREEN  = 900
        _WORLD   = 100.0
        _PADDING = 60   # pixels of margin on each side

        flat_ray_dim  = int(np.prod(self._ray_dim))
        rays = obs[:flat_ray_dim].reshape(self._ray_dim)

        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((_SCREEN, _SCREEN))
            pygame.display.set_caption(f"{title}")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        scale = (_SCREEN - 2 * _PADDING) / _WORLD

        self.screen.fill((255, 255, 255))

        for start, end in self._env.walls:
            pygame.draw.line(self.screen, (0, 0, 0),
                              w2s(start, scale, _SCREEN, _PADDING), 
                              w2s(end, scale, _SCREEN, _PADDING), 2)

        pygame.draw.circle(self.screen, (0, 100, 255),
                            w2s(self._env.agent_pos[0], scale, _SCREEN, _PADDING), 6)
        pygame.draw.circle(self.screen, (0, 255, 0),
                            w2s(self._env.goal_pos, scale, _SCREEN, _PADDING), 8)

        intersect, _, _ = self._env.ray_cast.scan(self._env.agent_pos, self._env.facing_direction)
        agent_screen = w2s(self._env.agent_pos[0], scale, _SCREEN, _PADDING)
        for i,ray in enumerate(intersect[0]):   # env 0 rays: (num_rays, 2)
            color = rays[4:, i]
            pygame.draw.line(self.screen, (int(color[0] * 255), 
                                           int(color[1] * 255), 
                                           int(color[2] * 255)),
                                             agent_screen, w2s(ray, scale, _SCREEN, _PADDING), 1)

        pygame.display.flip()
        self.clock.tick(5)


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
