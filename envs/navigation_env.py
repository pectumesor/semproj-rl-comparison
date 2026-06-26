import torch
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig
from .env_utils import RayCast, walls_json_to_numpy, compute_starts_and_ends, PerlinColor

class NavigationEnv(gym.Env):
    """
    Vectorized navigation environment.

    All state is stored as on-device tensors. reset() and step() return tensors.

    Observation per env: (7, num_rays)
        channel 0 — no_hit indicator
        channel 1 — goal_hit indicator  (ray-circle test; not a wall cast)
        channel 2 — wall_hit indicator
        channel 3 — normalised distance [0, 1]
        channels 4-6 — Perlin RGB at the hit / endpoint global (x, y)

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

        self.color_field = PerlinColor(device=device)

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
        # Proprioceptive state — last normalised action values, reset to 0 at episode start
        self.last_speed   = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.last_turning = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Dict observation space: structured ray matrix + flat proprio vector
        self.observation_space = gym.spaces.Dict({
            "rays":    gym.spaces.Box(0.0, 1.0,  shape=self.obs_dim, dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=(2,),          dtype=np.float32),
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
            low=-1.0,
            high=1.0,
            dtype=np.float32,
            shape=(self.act_dim,)
        )

    def compile(self, mode: str = "reduce-overhead"):
        """Fuse hot-path kernels with torch.compile. Call once after construction."""
        self.ray_cast.scan   = torch.compile(self.ray_cast.scan,   mode=mode)
        self.get_observations = torch.compile(self.get_observations, mode=mode)
        self.color_field      = torch.compile(self.color_field,      mode=mode)
        return self

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
        self.last_speed[mask]       = 0.0
        self.last_turning[mask]     = 0.0

        intersections, distances, d_unit = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        self.prev_dist = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        return self.get_observations(intersections, distances, d_unit), {}

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

        dx    = speed * torch.cos(self.facing_direction)
        dy    = speed * torch.sin(self.facing_direction)
        delta = torch.stack([dx, dy], dim=-1)           # (E, 2)

        # Reuse intersect() — treat movement as a single ray per env
        min_t  = self.ray_cast.intersect(self.agent_pos, delta[:, None, :]).squeeze(1)  # (E,)
        safe_t = (min_t - 1e-3).clamp(0.0, 1.0)
        self.agent_pos        += delta * safe_t[:, None]
        self.facing_direction += turning
        self.last_speed        = action[:, 1]
        self.last_turning      = action[:, 0]

        intersections, distances, d_unit = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        obs = self.get_observations(intersections, distances, d_unit)

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
        return obs, reward, terminated, truncated, {
            "agent_pos": self.agent_pos,
            "last_turning": action[:,0],
            "last_speed": action[:,1]
        }

    def get_observations(
        self, intersections: torch.Tensor, distances: torch.Tensor, d_unit: torch.Tensor
    ) -> torch.Tensor:
        max_range = self.ray_cast.max_range

        no_hit   = torch.isinf(distances)
        wall_hit = ~no_hit

        # Ray-circle test for goal: rays are not cast against the goal as geometry,
        # so we project each ray onto the goal disk directly.
        to_goal = self.goal_pos[None, None, :] - self.agent_pos[:, None, :]            # (E, 1, 2)
        t_goal = (to_goal * d_unit).sum(dim=-1).clamp(0.0, max_range)                  # (E, R)
        closest = self.agent_pos[:, None, :] + t_goal[:, :, None] * d_unit            # (E, R, 2)
        dist_sq_to_goal = ((closest - self.goal_pos[None, None, :]) ** 2).sum(dim=-1) # (E, R)
        wall_dist = torch.where(no_hit, torch.full_like(distances, max_range), distances)
        goal_hit = (dist_sq_to_goal <= self.goal_radius ** 2) & (t_goal < wall_dist)
        wall_hit = wall_hit & ~goal_hit

        rays = torch.zeros(self.num_envs, 7, self.num_rays,
                           dtype=torch.float32, device=self.device)
        rays[:, 0, :] = no_hit.float()
        rays[:, 1, :] = goal_hit.float()
        rays[:, 2, :] = wall_hit.float()
        rays[:, 3, :] = torch.where(no_hit, torch.ones_like(distances), distances / max_range)

        goal_pos_exp = self.goal_pos[None, None, :].expand(self.num_envs, self.num_rays, 2)
        sample_pts = torch.where(goal_hit[:, :, None], goal_pos_exp, intersections)
        rgb = self.color_field(sample_pts[:, :, 0], sample_pts[:, :, 1])
        rays[:, 4:, :] = rgb.permute(0, 2, 1)

        # Proprioceptive: vestibular-style rates only (no absolute heading — humans don't have a compass)
        proprio = torch.stack([
            self.last_speed,
            self.last_turning,
        ], dim=-1)  # (E, 2)

        return {"rays": rays, "proprio": proprio}

    def render(self):
        pass

class NavigationEnvEasy(NavigationEnv):
    """
    Wrapper to NavigationEnv that enhances the observations to make learning easy.
    Used to compare custom PPO/SAC implementations with Stable Baselines3
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
        
        super().__init__(cfg, agent, num_rays, obs_dim, num_envs, device)
       
        # Dict observation space: structured ray matrix + flat proprio vector
        self.observation_space = gym.spaces.Dict({
            "rays":    gym.spaces.Box(0.0, 1.0,  shape=self.obs_dim, dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=(4,),          dtype=np.float32),
        })

    def get_observations(
        self, intersections: torch.Tensor, distances: torch.Tensor, d_unit: torch.Tensor
    ) -> torch.Tensor:
        max_range = self.ray_cast.max_range

        no_hit   = torch.isinf(distances)
        wall_hit = ~no_hit

        # Ray-circle test for goal: rays are not cast against the goal as geometry,
        # so we project each ray onto the goal disk directly.
        to_goal = self.goal_pos[None, None, :] - self.agent_pos[:, None, :]            # (E, 1, 2)
        t_goal = (to_goal * d_unit).sum(dim=-1).clamp(0.0, max_range)                  # (E, R)
        closest = self.agent_pos[:, None, :] + t_goal[:, :, None] * d_unit            # (E, R, 2)
        dist_sq_to_goal = ((closest - self.goal_pos[None, None, :]) ** 2).sum(dim=-1) # (E, R)
        wall_dist = torch.where(no_hit, torch.full_like(distances, max_range), distances)
        goal_hit = (dist_sq_to_goal <= self.goal_radius ** 2) & (t_goal < wall_dist)
        wall_hit = wall_hit & ~goal_hit

        rays = torch.zeros(self.num_envs, 7, self.num_rays,
                           dtype=torch.float32, device=self.device)
        rays[:, 0, :] = no_hit.float()
        rays[:, 1, :] = goal_hit.float()
        rays[:, 2, :] = wall_hit.float()
        rays[:, 3, :] = torch.where(no_hit, torch.ones_like(distances), distances / max_range)

        goal_pos_exp = self.goal_pos[None, None, :].expand(self.num_envs, self.num_rays, 2)
        sample_pts = torch.where(goal_hit[:, :, None], goal_pos_exp, intersections)
        rgb = self.color_field(sample_pts[:, :, 0], sample_pts[:, :, 1])
        rays[:, 4:, :] = rgb.permute(0, 2, 1)

        # Proprioceptive:
        proprio = torch.stack([
               torch.sin(self.facing_direction),
               torch.cos(self.facing_direction),
            self.last_speed,
            self.last_turning,
           ], dim=-1)
        return {"rays": rays, "proprio": proprio}

