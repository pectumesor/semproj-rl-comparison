"""
Simplified version of NavigationEnv used for trajectory generation for Grid Cell Network Pre-Training
"""

import io
import torch
import numpy as np
import gymnasium as gym
import pygame
import imageio.v3 as iio
from omegaconf import DictConfig
from .env_utils import (RayCast, walls_json_to_numpy, compute_starts_and_ends,
                         PerlinColor, w2s, bounding_box, diagonal_length)

class TrajGenEnv(gym.Env):
    """
    Vectorized navigation environment.

    All state is stored as on-device tensors. reset() and step() return tensors.

    Observation rays: (num_envs, 1, num_rays)
        channel 1  — normalised distance [0, 1]
    
    Observatio proprioceptive: (num_envs, 3)
        - Absolute position
        - Absolute head direction
        
    Action: (num_envs, 2)
        [:, 0] turning in [-1, 1]  →  ±half_fov radians
        [:, 1] speed   in [-1, 1]  →  ±max_speed
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_rays: int,
        obs_dim: tuple,
        num_envs: int,
        device: str = "cpu",
    ):
        self.obs_dim   = obs_dim
        self.act_dim   = cfg.env.act_dim
        self.max_speed = cfg.env.max_speed
        self.fov       = cfg.env.fov
        self.num_rays  = num_rays
        self.num_envs  = num_envs
        self.device    = device

        self._half_fov_rad = float(np.deg2rad(self.fov / 2.0))

        self.walls = walls_json_to_numpy(cfg.env.room_path)
        ws_np, we_np = compute_starts_and_ends(self.walls)
        self.bounding_box = [*bounding_box(we_np, ws_np)]
        if cfg.env.range_type == "diagonal":
            max_range = diagonal_length(*self.bounding_box)
        elif cfg.env.range_type == "horizontal":
            max_range = self.bounding_box[1] - self.bounding_box[0]
        else:
            max_range = torch.inf

        wall_starts = torch.tensor(ws_np, dtype=torch.float32, device=device)
        wall_ends   = torch.tensor(we_np, dtype=torch.float32, device=device)
        self.ray_cast = RayCast(cfg, wall_starts, wall_ends, num_rays, max_range).to(device)

        # Mutable state
        self.agent_pos        = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        self.facing_direction = torch.zeros(num_envs,    dtype=torch.float32, device=device)

        # Dict observation space: structured ray matrix + flat proprio vector
        self.observation_space = gym.spaces.Dict({
            "rays":    gym.spaces.Box(0.0, 1.0,  shape=self.obs_dim, dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=(3,),          dtype=np.float32),
        })

        """
         
        -- Action Space --

        1st Dimension: 
            - Turning relative to facing direction.
            - Normalized to [-1, 1]
            - -1: left limit of field of view. +1 right limit
        2nd Dimension:
            - Forward velocity.
            - Normalized to [0, 1]
            - 1 Max speed
        """

        self.action_space = gym.spaces.Box(
            low=cfg.env.action_low,
            high=cfg.env.action_high,
            dtype=np.float32,
            shape=(self.act_dim,)
        )

    def compile(self, mode: str = "reduce-overhead"):
        """Fuse hot-path kernels with torch.compile. Call once after construction."""
        self.ray_cast.scan   = torch.compile(self.ray_cast.scan,   mode=mode)
        self.get_observations = torch.compile(self.get_observations, mode=mode)
        return self

    def reset(self, seed=None, options=None):
        """
        done: bool tensor (num_envs,) — reset only those envs. None resets all.
        Returns (obs, {}) where obs is (num_envs, 4, num_rays) on device.
        """
        
        eps = 1.0
        x = torch.empty((self.num_envs,), device=self.device).uniform_(self.bounding_box[0] + eps, self.bounding_box[1] - eps)
        y = torch.empty((self.num_envs,), device=self.device).uniform_(self.bounding_box[2] + eps, self.bounding_box[3] - eps)
        rand_pos = torch.stack([x,y], dim=-1)

        self.agent_pos        = rand_pos
        self.facing_direction = torch.empty((self.num_envs,), device=self.device).uniform_(0, 2 * torch.pi)
        _, distances, _ = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        return self.get_observations(distances), {}

    def step(self, action: torch.Tensor):
        """
        action: (num_envs, 2) tensor on device.
        Returns obs, reward, terminated, truncated, info — all tensors on device.
        """

        turning = action[:, 0] * self._half_fov_rad
        speed   = action[:, 1] * self.max_speed

        dx    = speed * torch.cos(self.facing_direction)
        dy    = speed * torch.sin(self.facing_direction)
        delta = torch.stack([dx, dy], dim=-1)           # (E, 2)

        # Reuse intersect() — treat movement as a single ray per env
        min_t  = self.ray_cast.intersect(self.agent_pos, delta[:, None, :]).squeeze(1)  # (E,)
        safe_t = (min_t - 1e-3).clamp(0.0, 1.0)
        self.agent_pos        += delta * safe_t[:, None]
        self.facing_direction += turning

        _, distances, _ = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        obs = self.get_observations(distances)

        return obs, {
            "last_turning": action[:,0],
            "last_speed": action[:,1]
        }

    def get_observations(
        self, distances: torch.Tensor
    ) -> torch.Tensor:
        max_range = self.ray_cast.max_range
        no_hit   = torch.isinf(distances)

        rays = torch.zeros(self.num_envs, 1, self.num_rays,
                           dtype=torch.float32, device=self.device)
       
        rays[:, 0, :] = torch.where(no_hit, torch.ones_like(distances), distances / max_range)

        proprio = torch.cat([
            self.agent_pos,
            self.facing_direction[:, None],
        ], dim=-1)  # (E, 3)

        return {"rays": rays, "proprio": proprio}
