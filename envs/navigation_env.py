import torch
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig
from .env_utils import RayCast, walls_json_to_numpy, compute_starts_and_ends



class NavigationEnv(gym.Env):
    """
    Vectorized navigation environment.

    All state is stored as on-device tensors. reset() and step() return tensors.

    Observation per env: (4, num_rays)
        channel 0 — no_hit indicator
        channel 1 — goal_hit indicator
        channel 2 — wall_hit indicator
        channel 3 — normalised distance [0, 1]

    Action: (num_envs, 2)
        [:, 0] turning in [-1, 1]  →  ±half_fov radians
        [:, 1] speed   in [-1, 1]  →  ±max_speed
    """

    def __init__(
        self,
        cfg: DictConfig,
        agent,
        num_rays: int,
        obs_dim: tuple,
        num_envs: int,
        device: str = "cpu",
    ):
        self.obs_dim   = obs_dim
        self.act_dim   = cfg.env.act_dim
        self.max_speed = cfg.env.max_speed
        self.fov       = cfg.env.fov
        self.max_steps = cfg.env.max_steps
        self.num_rays  = num_rays
        self.num_envs  = num_envs
        self.device    = device
        self.agent     = agent

        self._half_fov_rad = float(np.deg2rad(self.fov / 2.0))
        self.goal_radius  = float(cfg.env.get("goal_radius", 1e-4))
        self.dense_reward = bool(cfg.env.get("dense_reward", False))

        self.initial_pos = torch.tensor(
            [cfg.env.init_pos["x"], cfg.env.init_pos["y"]],
            dtype=torch.float32, device=device,
        )
        self.goal_pos = torch.tensor(
            [cfg.env.goal_pos["x"], cfg.env.goal_pos["y"]],
            dtype=torch.float32, device=device,
        )

        walls = walls_json_to_numpy(cfg.env.room_path)
        ws_np, we_np = compute_starts_and_ends(walls)
        wall_starts = torch.tensor(ws_np, dtype=torch.float32, device=device)
        wall_ends   = torch.tensor(we_np, dtype=torch.float32, device=device)
        self.ray_cast = RayCast(cfg, wall_starts, wall_ends, num_rays).to(device)

        # Mutable state
        self.agent_pos        = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        self.facing_direction = torch.zeros(num_envs,    dtype=torch.float32, device=device)
        self.steps            = torch.zeros(num_envs,    dtype=torch.long,    device=device)
        self.prev_dist        = torch.zeros(num_envs,    dtype=torch.float32, device=device)

        """
        
        -- Observation Space --

        A matrix of dimension (num_classes + 1) x num_rays

        Each column is a one-hot encodded vector of size: num_classes.
        The last entry of a column is a normalized distance of how far the ray was cast
        
        """

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            dtype=np.float32,
            shape=self.obs_dim
        )

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
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )


    def reset(self, seed=None, options=None, done: torch.Tensor = None):
        """
        done: bool tensor (num_envs,) — reset only those envs. None resets all.
        Returns (obs, {}) where obs is (num_envs, 4, num_rays) on device.
        """
        mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if done is None else done)

        self.agent_pos[mask]        = self.initial_pos
        self.facing_direction[mask] = np.pi / 2
        self.steps[mask]            = 0

        intersections, distances = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        self.prev_dist = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        return self.get_observations(intersections, distances), {}


    def step(self, action: torch.Tensor):
        """
        action: (num_envs, 2) tensor on device.
        Returns obs, reward, terminated, truncated, info — all tensors on device.
        """
        self.steps += 1
        truncated = self.steps >= self.max_steps
        self.steps[truncated] = 0

        turning = action[:, 0] * self._half_fov_rad
        speed   = action[:, 1] * self.max_speed

        dx = speed * torch.cos(self.facing_direction)
        dy = speed * torch.sin(self.facing_direction)
        self.agent_pos        += torch.stack([dx, dy], dim=-1)
        self.facing_direction += turning

        intersections, distances = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        obs = self.get_observations(intersections, distances)

        dist_to_goal = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        terminated = dist_to_goal <= self.goal_radius

        reward = torch.full(
            (self.num_envs,), -1.0 / self.max_steps,
            dtype=torch.float32, device=self.device,
        )
        if self.dense_reward:
            reward += (self.prev_dist - dist_to_goal) / (self.ray_cast.max_range * self.max_steps)
        reward[terminated] += 1.0

        self.prev_dist = dist_to_goal
        return obs, reward, terminated, truncated, {}


    def get_observations(
        self, intersections: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        max_range = self.ray_cast.max_range

        no_hit   = torch.isinf(distances)
        hit      = ~no_hit
        dist_to_goal = torch.norm(intersections - self.goal_pos[None, None, :], dim=-1)
        goal_hit = hit & (dist_to_goal <= 1e-4)
        wall_hit = hit & ~goal_hit

        obs = torch.zeros(self.num_envs, 4, self.num_rays,
                          dtype=torch.float32, device=self.device)
        obs[:, 0, :] = no_hit.float()
        obs[:, 1, :] = goal_hit.float()
        obs[:, 2, :] = wall_hit.float()
        obs[:, 3, :] = torch.where(no_hit, torch.ones_like(distances), distances / max_range)

        return obs
