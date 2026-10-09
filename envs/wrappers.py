import io
import gymnasium as gym
import numpy as np
import torch
import pygame
import imageio.v3 as iio
from omegaconf import DictConfig
from .navigation_env import NavigationEnvEasy, NavigationEnv
from models.embeddings.simple import MLPObservationEmbeddings
from models.backbones.mlp_backbone import MLPBackbone
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils.geometry import w2s


class FrameStackWrapper(gym.Wrapper):
    """
    Stacks the last `stack_size` observations along a new axis placed right after
    num_envs, i.e. rays: (num_envs, stack_size, C, R), proprio: (num_envs, stack_size, P).

    gymnasium's built-in FrameStackObservation doesn't fit here: NavigationEnv is
    already batched over num_envs (every obs/action carries a leading num_envs dim
    that isn't reflected in its single-instance observation_space), returns torch
    tensors instead of numpy, and its reset() takes a `done` mask to reset only
    the finished sub-envs rather than gymnasium's per-episode autoreset. This wrapper
    mirrors that batched-torch, done-masked-reset API instead of gymnasium's.
    """

    def __init__(self, env, stack_size: int):
        super().__init__(env)

        self.stack_size = stack_size
        self.num_envs = env.num_envs
        self.device = env.device
        self._stack = None

        self.observation_space = gym.spaces.Dict({
            key: gym.spaces.Box(space.low.min(), space.high.max(),
                                shape=(stack_size, *space.shape), dtype=space.dtype)
            for key, space in env.observation_space.spaces.items()
        })

    def _allocate_stack(self, raw_obs: dict):
        self._stack = {
            key: torch.zeros(self.num_envs, self.stack_size, *val.shape[1:],
                             dtype=val.dtype, device=val.device)
            for key, val in raw_obs.items()
        }

    def reset(self, seed=None, options=None, done: torch.Tensor = None):
        mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if done is None else done)

        raw_obs, info = self.env.reset(seed=seed, options=options, done=done)

        if self._stack is None:
            self._allocate_stack(raw_obs)

        for key, val in raw_obs.items():
            fresh = val[mask].unsqueeze(1).expand(-1, self.stack_size, *val.shape[1:])
            self._stack[key][mask] = fresh

        return {key: val.clone() for key, val in self._stack.items()}, info

    def step(self, action: torch.Tensor):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)

        for key, val in raw_obs.items():
            self._stack[key] = torch.cat([self._stack[key][:, 1:], val.unsqueeze(1)], dim=1)

        return {key: val.clone() for key, val in self._stack.items()}, reward, terminated, truncated, info

    def record_rollout(self, backbone_type, agent, steps, cfg: DictConfig):
        if backbone_type == "lstm":
            return self.record_recurrent_rollout(agent, cfg.backbone.lstm_num_layers,
                                                  cfg.backbone.lstm_backbone_feature_dim, steps)
        else:
            return self.record_mlp_rollout(agent, steps)
        
    @torch.inference_mode()
    def record_mlp_rollout(self, agent, steps):

        frames = []
        obs, _ = self.reset()
        for _ in range(steps):
            action = agent.predict_action(obs)
            obs, _, done, _, _ = self.step(action)
            last_ray = obs["rays"][:, -1]
            last_proprio = obs["proprio"][:, -1]
            record_obs = {"rays": last_ray,
                          "proprio": last_proprio}
            
            frames.append(self.env.record_frame(record_obs))

            if done.any():
                obs, _ = self.reset()

        return frames
    
    @torch.inference_mode()
    def record_recurrent_rollout(self, agent, num_layers, hidden_size, steps):

        frames = []
        lstm_state = (
            torch.zeros((num_layers, 1, hidden_size),
                        dtype=torch.float, device=self.device),
            torch.zeros((num_layers, 1, hidden_size),
                         dtype=torch.float, device=self.device)
            )
        done = torch.zeros(1, dtype=torch.bool, device=self.device)
        obs, _ = self.reset()
        for _ in range(steps):
            action, lstm_state = agent.predict_action(obs, lstm_state, done)
            obs, _, done, _, _ = self.step(action)
            last_ray = obs["rays"][:, -1]
            last_proprio = obs["proprio"][:, -1]
            record_obs = {"rays": last_ray,
                          "proprio": last_proprio}
            frames.append(self.env.record_frame(record_obs))

            if done.any():
                obs, _ = self.reset()

        return frames


class NavigationEnvSB3(gym.Env):
    """
    Single-env wrapper around NavigationEnv for Stable Baselines 3.
    """

    def __init__(self, cfg, num_rays: int, ray_dim: tuple, proprio_dim: int, device: str = "cpu"):
        super().__init__()
        self._env       = NavigationEnv(cfg, None, num_rays, ray_dim, num_envs=1, device=device)
        self._ray_dim   = ray_dim
        self._proprio_dim = proprio_dim
        flat_dim = int(np.prod(ray_dim)) + proprio_dim

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(flat_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.asarray(cfg.env.action_low, dtype=np.float32),
            high=np.asarray(cfg.env.action_high, dtype=np.float32), dtype=np.float32)

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

    def record_frame(self, obs: np.ndarray) -> np.ndarray:
        _SCREEN  = 900
        _WORLD   = 100.0
        _PADDING = 60

        flat_ray_dim = int(np.prod(self._ray_dim))
        rays = obs[:flat_ray_dim].reshape(self._ray_dim)

        if not hasattr(self, '_rec_screen'):
            pygame.init()
            self._rec_screen = pygame.Surface((_SCREEN, _SCREEN))

        scale = (_SCREEN - 2 * _PADDING) / _WORLD
        self._rec_screen.fill((255, 255, 255))

        for start, end in self._env.walls:
            pygame.draw.line(self._rec_screen, (0, 0, 0),
                             w2s(start, scale, _SCREEN, _PADDING),
                             w2s(end,   scale, _SCREEN, _PADDING), 2)

        pygame.draw.circle(self._rec_screen, (0, 100, 255),
                           w2s(self._env.agent_pos[0], scale, _SCREEN, _PADDING), 6)
        pygame.draw.circle(self._rec_screen, (0, 255, 0),
                           w2s(self._env.goal_pos, scale, _SCREEN, _PADDING), 8)

        intersect, _, _ = self._env.ray_cast.scan(self._env.agent_pos, self._env.facing_direction)
        agent_screen = w2s(self._env.agent_pos[0], scale, _SCREEN, _PADDING)
        for i, ray in enumerate(intersect[0]):
            color = rays[4:, i]
            pygame.draw.line(self._rec_screen,
                             (int(color[0]*255), int(color[1]*255), int(color[2]*255)),
                             agent_screen, w2s(ray, scale, _SCREEN, _PADDING), 2)

        frame = np.transpose(pygame.surfarray.array3d(self._rec_screen), (1,0,2))
        return frame

    def record_rollout(self, agent, steps):
        frames = []
        obs, _ = self.reset()
        for _ in range(steps):
            action, _ = agent.predict(obs)
            obs, _, done, _, _ = self.step(action)
            frames.append(self.record_frame(obs))

        if done:
            obs, _ = self.reset()
        
        return frames
    
    def record_recurrent_rollout(self, agent, num_layers, batch_size, hidden_size, steps, device):

        frames = []
        lstm_state = (
            torch.zeros((num_layers, batch_size, hidden_size),
                        dtype=torch.float, device=device),
            torch.zeros((num_layers, batch_size, hidden_size),
                         dtype=torch.float, device=device)
            )
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        obs, _ = self.reset()
        for _ in range(steps):
            action, lstm_state = agent.predict(obs, lstm_state, done)
            obs, _, done, _, _ = self.step(action)
            frames.append(self.record_frame(obs))

            if done:
                obs, _ = self.reset()

        return frames

class NavigationEnvEasySB3(NavigationEnvSB3):
    def __init__(self, cfg, num_rays, ray_dim, proprio_dim, device = "cpu"):
        super().__init__(cfg, num_rays, ray_dim, proprio_dim, device)

        self._env = NavigationEnvEasy(cfg, None, num_rays, ray_dim, num_envs=1, device=device)

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
