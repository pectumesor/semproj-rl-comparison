import io
import torch
import numpy as np
import gymnasium as gym
import pygame
import imageio.v3 as iio
from omegaconf import DictConfig
from .env_utils import RayCast, PerlinColor, GridCode, RandomFourierFeatures, information_dims
from utils.geometry import (walls_json_to_numpy, compute_starts_and_ends,
                             w2s, bounding_box, diagonal_length, start_goal_pos_from_json, InitialPositionSampler,
                             wall_attributes_json, goal_regions_json, junction_region_json, junction_path_json)

WALL_OFFSET = 10
# Cue colours used when a room has cue walls but no "cue_colours" field
DEFAULT_CUE_COLOURS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]

class NavigationEnv(gym.Env):
    """
    Vectorized navigation environment.

    All state is stored as on-device tensors. reset() and step() return tensors.

    Observation rays: (num_envs, 7, num_rays)
        channel 0 — no_hit indicator
        channel 1 — goal_hit indicator  (ray-circle test; not a wall cast)
        channel 2 — wall_hit indicator
        channel 3 — normalised distance [0, 1]
        channels 4-6 — RGB of the hit point, depending on env.colour_mode:
            global     — Perlin field over global (x, y): acts as a place code
            room_local — Perlin field over (x, y) mod colour_period: identical rooms on that grid look identical
            off        — black, the agent only sees geometry
            per_wall   — "colour" of the hit edge in the room JSON
          Cue walls ("cue": true in the room JSON) always show the episode's cue colour.

    Observatio proprioceptive: (num_envs, proprio_dim)
        - last steps speed
        - last steps turning angle
        - information slots (env.information), see information_dims():
            goal      (5)  distance, cos/sin of the egocentric bearing, seen flag, age
            position  (G)  xy / random Fourier features / grid code / grid code of a noisy path integration
            goal grid (G)  grid code of the goal position


    Action: (num_envs, 2)
        [:, 0] turning in [-1, 1]  →  ±max_turn_deg
        [:, 1] speed   in [-1, 1]  →  ±max_speed

    Episodes terminate on success (within goal_radius of the goal) or on failure (entering a goal
    region that isn't the episode's goal, goal_mode=regions). info["success"] tells the two apart.
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
        self.max_steps = cfg.algorithm.max_steps
        self.num_rays  = num_rays
        self.num_envs  = num_envs
        self.device    = device
        self.agent     = agent
        self.random_pos_flag = cfg.env.rand_pos == "True"
        self.seed = cfg.seed

        self._half_fov_rad = float(np.deg2rad(self.fov / 2.0))
        # Turning used to be scaled by half the FOV, so an FOV sweep also changed the action space.
        # max_turn_deg decouples them (90 = old behaviour at the default 180° FOV)
        self._max_turn_rad = float(np.deg2rad(cfg.env.get("max_turn_deg", self.fov / 2.0)))
        self.goal_radius  = float(cfg.env.get("goal_radius", 1e-4))                        # success threshold
        self.goal_visual_radius = float(cfg.env.get("goal_visual_radius", self.goal_radius))  # disk the rays can see
        self.collision_penalty = float(cfg.env.get("collision_penalty", 0.0))

        start_pos, goal_pos = start_goal_pos_from_json(cfg.env.room_path)
        self.spawn_sampler = InitialPositionSampler(cfg.env.room_path, wall_offset=WALL_OFFSET, device=device)

        # -- Goals: one per env --
        self.goal_mode = cfg.env.get("goal_mode", "fixed")
        self.min_goal_distance = float(cfg.env.get("min_goal_distance", 0.0))
        self.fixed_goal = torch.tensor(goal_pos, dtype=torch.float32, device=device)
        self.goal_pos = self.fixed_goal.expand(num_envs, 2).clone()           # (E, 2)
        self.goal_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        goal_layout = goal_regions_json(cfg.env.room_path)
        self.goal_points, self.goal_boxes = None, None
        if self.goal_mode == "regions":
            if goal_layout is None:
                raise ValueError(f"goal_mode=regions needs goal_points in {cfg.env.room_path}")
            points, boxes, _ = goal_layout
            self.goal_points = torch.tensor(points, device=device)            # (K, 2)
            self.goal_boxes = torch.tensor(boxes, device=device) if boxes is not None else None  # (K, 4)
        # Optional junction of the twin rooms, only read by the E2 diagnostics (scripts/e2_diagnostics.py)
        junction = junction_region_json(cfg.env.room_path)
        self.junction_box = torch.tensor(junction, device=device) if junction is not None else None  # (4,)
        path = junction_path_json(cfg.env.room_path)
        self.junction_path = torch.tensor(path, device=device) if path is not None else None  # (P, 2)
        self.past_junction = torch.zeros(num_envs, dtype=torch.bool, device=device)  # entered the junction this episode

        # Potential-based shaping toward the junction (twin rooms, E2), see configs/env/base_env.yaml
        self.junction_shaping = float(cfg.env.get("junction_shaping", 0.0))
        self.gamma = float(cfg.env.gamma)
        self.junction_potential = torch.zeros(num_envs, dtype=torch.float32, device=device)  # Φ(s_t)
        if self.junction_shaping != 0.0 and (self.junction_path is None or self.junction_box is None):
            raise ValueError(f"env.junction_shaping needs junction_path and junction_region in {cfg.env.room_path}")

        # -- Colours --
        self.colour_mode = cfg.env.get("colour_mode", "global")
        self.colour_period = float(cfg.env.get("colour_period", 200.0))
        # Fixed across RL seeds by default, so every seed sees the same colour map
        colour_seed = cfg.env.get("colour_seed", None)
        self.color_field = PerlinColor(device=device, seed=cfg.seed if colour_seed is None else colour_seed)

        wall_colours, cue_walls = wall_attributes_json(cfg.env.room_path)
        self.wall_colours = torch.tensor(wall_colours, device=device)         # (W, 3)
        self.cue_walls = torch.tensor(cue_walls, device=device)               # (W,)
        cue_colours = goal_layout[2] if goal_layout is not None and goal_layout[2] is not None else DEFAULT_CUE_COLOURS
        self.cue_colours = torch.tensor(np.asarray(cue_colours, dtype=np.float32), device=device)  # (K, 3)
        self.cue_colour = self.cue_colours[0].expand(num_envs, 3).clone()    # (E, 3)
        self.cue_visible = torch.zeros(num_envs, dtype=torch.bool, device=device)  # any ray on a cue wall this step

        self.walls = walls_json_to_numpy(cfg.env.room_path)
        ws_np, we_np = compute_starts_and_ends(self.walls)
        self.bounding_box = self.spawn_sampler.bounding_box
        self.diagonal = diagonal_length(*self.bounding_box)
        if cfg.env.range_type == "diagonal":
            max_range = self.diagonal
        elif cfg.env.range_type == "horizontal":
            max_range = float(self.bounding_box[1] - self.bounding_box[0])
        else:
            max_range = torch.inf

        wall_starts = torch.tensor(ws_np, dtype=torch.float32, device=device)
        wall_ends   = torch.tensor(we_np, dtype=torch.float32, device=device)
        self.ray_cast = RayCast(cfg, wall_starts, wall_ends, num_rays, max_range).to(device)

        self.agent_pos        = self.spawn_sampler.sample(self.num_envs)
        self.facing_direction = torch.empty((self.num_envs,), device=self.device).uniform_(0, 2 * torch.pi)

        self.initial_pos = torch.tensor(
                    start_pos,
                    dtype=torch.float32, device=device,
                )
        # Path distance from the start to the junction: normalises the potential, so the shaping adds up to
        # junction_shaping over the way to the junction at every corridor length L
        self.junction_norm = (float(self.junction_distance(self.initial_pos[None])[0])
                              if self.junction_path is not None else 1.0)

        self.steps            = torch.zeros(num_envs,    dtype=torch.long,    device=device)
        self.prev_dist        = torch.zeros(num_envs,    dtype=torch.float32, device=device)
        # Proprioceptive state — last normalised action values, reset to 0 at episode start
        self.last_speed   = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.last_turning = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # -- Information slots (goal / position / goal grid code) --
        info_cfg = cfg.env.information
        self.goal_info = info_cfg.goal_info
        self.position_info = info_cfg.position_info
        self.goal_grid = bool(info_cfg.goal_grid)
        self.goal_dim, self.position_dim, self.goal_grid_dim = information_dims(cfg)
        grid_cfg = info_cfg.grid
        self.grid_code = GridCode(modules=grid_cfg.modules, phases=grid_cfg.phases,
                                  min_spacing=grid_cfg.min_spacing, spacing_ratio=grid_cfg.spacing_ratio,
                                  seed=info_cfg.seed, device=device)
        self.fourier_features = RandomFourierFeatures(dim=self.grid_code.dim, min_spacing=grid_cfg.min_spacing,
                                                      max_spacing=grid_cfg.min_spacing * grid_cfg.spacing_ratio ** (grid_cfg.modules - 1),
                                                      seed=info_cfg.seed, device=device)
        self.speed_noise   = float(info_cfg.path_integration.speed_noise)
        self.heading_noise = float(np.deg2rad(info_cfg.path_integration.heading_noise_deg))

        self.goal_seen       = torch.zeros(num_envs, dtype=torch.bool,    device=device)
        self.goal_age        = torch.zeros(num_envs, dtype=torch.float32, device=device)  # steps since the goal was last seen
        self.last_seen_goal  = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)  # world position at last sighting
        self.frozen_goal     = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)  # egocentric (d, cos, sin) at last sighting
        self.integrated_pos  = self.agent_pos.clone()                                       # noisy path integration estimate

        base_proprio_dim = len(self._base_proprio())
        proprio_dim = base_proprio_dim + self.goal_dim + self.position_dim + self.goal_grid_dim

        # Dict observation space: structured ray matrix + flat proprio vector
        self.observation_space = gym.spaces.Dict({
            "rays":    gym.spaces.Box(0.0, 1.0,  shape=self.obs_dim, dtype=np.float32),
            "proprio": gym.spaces.Box(-1.0, 1.0, shape=(proprio_dim,), dtype=np.float32),
        })

        """

        -- Action Space --

        1st Dimension:
            - Turning relative to facing direction.
            - Normalized to [-1, 1]
            - -1: max_turn_deg to the left. +1 max_turn_deg to the right
        2nd Dimension:
            - Forward velocity.
            - Normalized to [0, 1]
            - 1 Max speed
        """

        self.action_space = gym.spaces.Box(
            low=np.asarray(cfg.env.action_low, dtype=np.float32),
            high=np.asarray(cfg.env.action_high, dtype=np.float32),
            dtype=np.float32,
            shape=(self.act_dim,)
        )
        self._action_low  = torch.as_tensor(self.action_space.low,  device=device)
        self._action_high = torch.as_tensor(self.action_space.high, device=device)



    def compute_reward(self, success, collided=None):
        reward = torch.full(
            (self.num_envs,), -1.0 / self.max_steps,
            dtype=torch.float32, device=self.device,
        )

        reward[success] += 1.0
        if collided is not None and self.collision_penalty != 0.0:
            reward[collided] -= self.collision_penalty
        return reward

    def _reset_goals(self, mask: torch.Tensor):
        """Pick the goal of every env in mask, after the agent positions are set."""
        if self.goal_mode == "random":
            ids = mask.nonzero(as_tuple=True)[0]
            goals = self.spawn_sampler.sample(len(ids))
            # Rejection sampling: keep start and goal at least min_goal_distance apart
            for _ in range(InitialPositionSampler.MAX_RESAMPLES):
                close = torch.norm(goals - self.agent_pos[ids], dim=-1) < self.min_goal_distance
                if not close.any():
                    break
                goals[close] = self.spawn_sampler.sample(int(close.sum()))
            self.goal_pos[ids] = goals
        elif self.goal_mode == "regions":
            idx = torch.randint(0, len(self.goal_points), (self.num_envs,), device=self.device)
            self.goal_idx[mask] = idx[mask]
            self.goal_pos[mask] = self.goal_points[idx[mask]]
            self.cue_colour[mask] = self.cue_colours[idx[mask] % len(self.cue_colours)]

    def _reset_information(self, mask: torch.Tensor):
        self.goal_seen[mask]      = False
        self.goal_age[mask]       = 0.0
        self.last_seen_goal[mask] = 0.0
        self.frozen_goal[mask]    = 0.0
        self.integrated_pos[mask] = self.agent_pos[mask]

    def _reset_junction(self, mask: torch.Tensor):
        """Junction flag and shaping potential of every env in mask, after the agent positions are set."""
        self.past_junction[mask] = False
        if self.junction_shaping != 0.0:
            self.junction_potential[mask] = self.junction_potential_now()[mask]

    def reset(self, seed=None, options=None, done: torch.Tensor = None):
        """
        done: bool tensor (num_envs,) — reset only those envs. None resets all.
        Returns (obs, {}) where obs is (num_envs, 4, num_rays) on device.
        """
        mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if done is None else done)

        rand_pos = self.spawn_sampler.sample(self.num_envs)
        rand_direction = torch.empty((self.num_envs,), device=self.device).uniform_(0, 2 * torch.pi)

        self.agent_pos[mask]        = rand_pos[mask] if self.random_pos_flag else self.initial_pos
        self.facing_direction[mask] = rand_direction[mask]
        self.steps[mask]            = 0
        self.last_speed[mask]       = 0.0
        self.last_turning[mask]     = 0.0
        self._reset_goals(mask)
        self._reset_information(mask)
        self._reset_junction(mask)

        intersections, distances, d_unit, wall_idx = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        self.prev_dist = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        return self.get_observations(intersections, distances, d_unit, wall_idx), {"agent_pos": self.agent_pos}

    def observe(self):
        """Observation of the current state without stepping, e.g. after the E2 diagnostics moved the agent."""
        intersections, distances, d_unit, wall_idx = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        return self.get_observations(intersections, distances, d_unit, wall_idx)

    def step(self, action: torch.Tensor):
        """
        action: (num_envs, 2) tensor on device.
        Returns obs, reward, terminated, truncated, info — all tensors on device.
        """
        self.steps += 1
        truncated = self.steps >= self.max_steps
        self.steps[truncated] = 0

        # Enforce per-dimension bounds here too, so no caller (e.g. SAC warm start) can walk backwards
        action = action.clamp(min=self._action_low, max=self._action_high)

        turning = action[:, 0] * self._max_turn_rad
        speed   = action[:, 1] * self.max_speed

        dx    = speed * torch.cos(self.facing_direction)
        dy    = speed * torch.sin(self.facing_direction)
        delta = torch.stack([dx, dy], dim=-1)           # (E, 2)

        # Reuse intersect() — treat movement as a single ray per env
        min_t  = self.ray_cast.intersect(self.agent_pos, delta[:, None, :]).squeeze(1)  # (E,)
        safe_t = (min_t - 1e-3).clamp(0.0, 1.0)
        collided = torch.isfinite(min_t) & (speed > 0)   # a wall blocked this step's movement

        # Noisy odometry for the path-integrated position (noisy_grid): integrate the distance
        # actually travelled along the heading, both corrupted by Gaussian noise
        travelled = speed * safe_t
        noisy_heading = self.facing_direction + self.heading_noise * torch.randn_like(travelled)
        noisy_travel  = travelled * (1.0 + self.speed_noise * torch.randn_like(travelled))
        self.integrated_pos += noisy_travel[:, None] * torch.stack([torch.cos(noisy_heading), torch.sin(noisy_heading)], dim=-1)

        self.agent_pos        += delta * safe_t[:, None]
        self.facing_direction += turning
        self.last_speed        = action[:, 1]
        self.last_turning      = action[:, 0]
        self.goal_age         += 1.0
        self.past_junction    |= self.in_junction()

        intersections, distances, d_unit, wall_idx = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        obs = self.get_observations(intersections, distances, d_unit, wall_idx)

        dist_to_goal = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        success = dist_to_goal <= self.goal_radius
        failure = self.entered_wrong_region() & ~success
        terminated = success | failure

        reward = self.compute_reward(success, collided)
        if self.junction_shaping != 0.0:
            # F = γΦ(s') − Φ(s). Terminal states lie in a room, past the junction, so their Φ is 0 anyway; time-limit
            # cutoffs keep Φ(s'), since PPO bootstraps them with V(s')
            potential = torch.where(terminated, torch.zeros_like(reward), self.junction_potential_now())
            reward += self.gamma * potential - self.junction_potential
            self.junction_potential = potential

        self.prev_dist = dist_to_goal
        return obs, reward, terminated, truncated, {
            "agent_pos": self.agent_pos,
            "last_turning": action[:,0],
            "last_speed": action[:,1],
            "success": success,
            "collided": collided,
            "room": self.goal_region_index(),   # goal region the agent is in, -1 for none (E2 confusion matrix)
        }

    def _inside_goal_regions(self) -> torch.Tensor:
        """(E, K) True where the agent is inside goal region k."""
        x, y = self.agent_pos[:, 0:1], self.agent_pos[:, 1:2]                         # (E, 1)
        b = self.goal_boxes                                                           # (K, 4)
        return (x >= b[:, 0]) & (x <= b[:, 1]) & (y >= b[:, 2]) & (y <= b[:, 3])       # (E, K)

    def entered_wrong_region(self) -> torch.Tensor:
        """(E,) True for envs inside a goal region that doesn't hold their goal (goal_mode=regions)."""
        if self.goal_boxes is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        inside = self._inside_goal_regions()
        own = torch.arange(inside.shape[1], device=self.device)[None, :] == self.goal_idx[:, None]
        return (inside & ~own).any(dim=-1)

    def goal_region_index(self) -> torch.Tensor:
        """(E,) index of the goal region the agent is in, -1 outside every region (or without goal regions)."""
        if self.goal_boxes is None:
            return torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        inside = self._inside_goal_regions()
        return torch.where(inside.any(dim=-1), inside.float().argmax(dim=-1), torch.full_like(self.goal_idx, -1))

    def in_junction(self) -> torch.Tensor:
        """(E,) True for envs inside the room's junction_region (twin rooms), False everywhere otherwise."""
        if self.junction_box is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        x, y, b = self.agent_pos[:, 0], self.agent_pos[:, 1], self.junction_box
        return (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])

    def junction_distance(self, pos: torch.Tensor = None) -> torch.Tensor:
        """
        (E,) path distance to the end of the room's junction_path (the junction's corridor-side edge): the arc
        length left from the projection onto the nearest path segment. In the straight corridor it ignores the
        lateral offset, so the shaping doesn't pull the agent to either wall (or to the arm of either room).
        """
        pos = self.agent_pos if pos is None else pos
        a, b = self.junction_path[:-1], self.junction_path[1:]                     # (S, 2)
        seg = b - a
        seg_len = seg.norm(dim=-1)                                                   # (S,)
        rel = pos[:, None, :] - a[None]                                              # (E, S, 2)
        t = ((rel * seg[None]).sum(dim=-1) / seg_len.pow(2)).clamp(0.0, 1.0)         # (E, S)
        offset = (rel - t[..., None] * seg[None]).norm(dim=-1)                       # (E, S)
        after = seg_len.flip(0).cumsum(0).flip(0) - seg_len                          # (S,) length of the later segments
        remaining = (1.0 - t) * seg_len + after                                      # (E, S)
        return remaining.gather(1, offset.argmin(dim=-1, keepdim=True)).squeeze(1)

    def corridor_fraction(self) -> torch.Tensor:
        """(E,) progress along the corridor (last junction_path segment): 0 at the bend, 1 at the junction, < 0 before."""
        return 1.0 - self.junction_distance() / (self.junction_path[-1] - self.junction_path[-2]).norm()

    def junction_potential_now(self) -> torch.Tensor:
        """
        (E,) Φ = -junction_shaping * d_J / d_J(start), with d_J the path distance to the junction. 0 from the first
        step in the junction on: both rooms lie past it, so the shaping carries nothing about which room is right.
        """
        potential = -self.junction_shaping * self.junction_distance() / self.junction_norm
        return torch.where(self.past_junction, torch.zeros_like(potential), potential)

    def ray_colours(self, sample_pts: torch.Tensor, wall_idx: torch.Tensor, goal_hit: torch.Tensor) -> torch.Tensor:
        """RGB (E, R, 3) of every ray's hit point, see the colour_mode list in the class docstring."""
        if self.colour_mode == "global":
            rgb = self.color_field(sample_pts[:, :, 0], sample_pts[:, :, 1]) # Sample Perlin Noise color for the ray intersections
        elif self.colour_mode == "room_local":
            local = torch.remainder(sample_pts, self.colour_period)
            rgb = self.color_field(local[:, :, 0], local[:, :, 1])
        elif self.colour_mode == "per_wall":
            rgb = self.wall_colours[wall_idx.clamp(min=0)]
            rgb = torch.where(goal_hit[:, :, None], torch.zeros_like(rgb), rgb)
        elif self.colour_mode == "off":
            rgb = torch.zeros((*wall_idx.shape, 3), dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown colour_mode {self.colour_mode}")

        # Cue walls show the episode's cue colour whatever the colour mode
        cue_hit = self.cue_walls[wall_idx.clamp(min=0)] & (wall_idx >= 0) & ~goal_hit
        self.cue_visible = cue_hit.any(dim=-1)
        return torch.where(cue_hit[:, :, None], self.cue_colour[:, None, :], rgb)

    def _egocentric(self, rel: torch.Tensor) -> torch.Tensor:
        """World offset (E, 2) -> (distance / diagonal, cos, sin) of the bearing relative to the heading."""
        dist = (torch.norm(rel, dim=-1) / self.diagonal).clamp(0.0, 1.0)
        bearing = torch.atan2(rel[:, 1], rel[:, 0]) - self.facing_direction
        return torch.stack([dist, torch.cos(bearing), torch.sin(bearing)], dim=-1)

    def update_goal_memory(self, goal_hit: torch.Tensor):
        """Remember where and how the goal looked the last time any ray saw it."""
        seen_now = goal_hit.any(dim=-1)
        self.last_seen_goal[seen_now] = self.goal_pos[seen_now]
        self.frozen_goal[seen_now]    = self._egocentric(self.goal_pos - self.agent_pos)[seen_now]
        self.goal_age[seen_now]       = 0.0
        self.goal_seen |= seen_now

    def goal_features(self) -> torch.Tensor:
        """(E, 5): distance, cos/sin of the egocentric goal bearing, seen flag, age since last seen."""
        if self.goal_info == "none":
            return torch.zeros(self.num_envs, 5, dtype=torch.float32, device=self.device)

        if self.goal_info == "oracle":       # as if the agent saw the goal through walls every step
            ego  = self._egocentric(self.goal_pos - self.agent_pos)
            seen = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            age  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        else:
            seen = self.goal_seen.float()
            age  = (self.goal_age / self.max_steps).clamp(0.0, 1.0) * seen
            if self.goal_info == "last_seen_updated":   # the env does the remembering: re-expressed in the current frame
                ego = self._egocentric(self.last_seen_goal - self.agent_pos)
            elif self.goal_info == "last_seen_frozen":  # as seen at the sighting, the agent must track its own motion
                ego = self.frozen_goal
            else:
                raise ValueError(f"Unknown goal_info {self.goal_info}")
            ego = ego * seen[:, None]

        return torch.cat([ego, seen[:, None], age[:, None]], dim=-1)

    def position_features(self) -> torch.Tensor:
        """(E, position_dim) position code, zero-padded to the reserved slot size."""
        if self.position_info == "none":
            feat = torch.zeros(self.num_envs, 0, dtype=torch.float32, device=self.device)
        elif self.position_info == "xy":
            min_x, max_x, min_y, max_y = self.bounding_box
            lo = torch.tensor([min_x, min_y], dtype=torch.float32, device=self.device)
            hi = torch.tensor([max_x, max_y], dtype=torch.float32, device=self.device)
            feat = (self.agent_pos - lo) / (hi - lo)
        elif self.position_info == "rff":
            feat = self.fourier_features(self.agent_pos)
        elif self.position_info == "grid":
            feat = self.grid_code(self.agent_pos)
        elif self.position_info == "noisy_grid":
            feat = self.grid_code(self.integrated_pos)
        else:
            raise ValueError(f"Unknown position_info {self.position_info}")

        pad = self.position_dim - feat.shape[-1]
        return torch.nn.functional.pad(feat, (0, pad))

    def information_features(self) -> list:
        feats = []
        if self.goal_dim:
            feats.append(self.goal_features())
        if self.position_dim:
            feats.append(self.position_features())
        if self.goal_grid_dim:
            feats.append(self.grid_code(self.goal_pos) if self.goal_grid
                         else torch.zeros(self.num_envs, self.goal_grid_dim, dtype=torch.float32, device=self.device))
        return feats

    def _base_proprio(self) -> list:
        return [self.last_speed, self.last_turning]

    def get_observations(
        self, intersections: torch.Tensor, distances: torch.Tensor, d_unit: torch.Tensor, wall_idx: torch.Tensor
    ) -> torch.Tensor:
        max_range = self.ray_cast.max_range

        no_hit   = torch.isinf(distances)
        wall_hit = ~no_hit
        wall_dist = torch.where(no_hit, torch.full_like(distances, max_range), distances)

        # Ray-circle test for goal: rays are not cast against the goal as geometry,
        # so we project each ray onto the goal disk directly.
        to_goal = self.goal_pos[:, None, :] - self.agent_pos[:, None, :]               # (E, 1, 2)
        t_goal = (to_goal * d_unit).sum(dim=-1).clamp(0.0, max_range)                  # (E, R)
        closest = self.agent_pos[:, None, :] + t_goal[:, :, None] * d_unit            # (E, R, 2)
        dist_sq_to_goal = ((closest - self.goal_pos[:, None, :]) ** 2).sum(dim=-1)    # (E, R)

        goal_hit = (dist_sq_to_goal <= self.goal_visual_radius ** 2) & (t_goal < wall_dist)
        wall_hit = wall_hit & ~goal_hit

        rays = torch.zeros(self.num_envs, 7, self.num_rays,
                           dtype=torch.float32, device=self.device)
        rays[:, 0, :] = no_hit.float()
        rays[:, 1, :] = goal_hit.float()
        rays[:, 2, :] = wall_hit.float()
        rays[:, 3, :] = torch.where(no_hit, torch.ones_like(distances), distances / max_range)

        goal_pos_exp = self.goal_pos[:, None, :].expand(self.num_envs, self.num_rays, 2)
        sample_pts = torch.where(goal_hit[:, :, None], goal_pos_exp, intersections)
        rgb = self.ray_colours(sample_pts, wall_idx, goal_hit)
        rgb = torch.where(no_hit[:, :, None], torch.zeros_like(rgb), rgb) # No wall or goal, Black
        rays[:, 4:, :] = rgb.permute(0, 2, 1)

        self.update_goal_memory(goal_hit)

        # Proprioceptive + information slots
        proprio = torch.cat([
            torch.stack(self._base_proprio(), dim=-1),
            *self.information_features(),
        ], dim=-1)  # (E, proprio_dim)

        return {"rays": rays, "proprio": proprio}

    def render(self, obs, title, mode="human"):

        _SCREEN  = 900
        _PADDING = 60   # pixels of margin on each side

        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((_SCREEN, _SCREEN))
            pygame.display.set_caption(f"{title}")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        min_x, max_x, min_y, max_y = self.bounding_box
        world = max(max_x - min_x, max_y - min_y)
        scale = (_SCREEN - 2 * _PADDING) / world

        self.screen.fill((255, 255, 255))

        for start, end in self.walls:
            pygame.draw.line(self.screen, (0, 0, 0),
                             w2s(start, scale, _SCREEN, _PADDING),
                               w2s(end, scale, _SCREEN, _PADDING), 2)

        pygame.draw.circle(self.screen, (0, 100, 255),
                            w2s(self.agent_pos[0], scale, _SCREEN, _PADDING), 6)
        pygame.draw.circle(self.screen, (0, 255, 0),
                              w2s(self.goal_pos[0], scale, _SCREEN, _PADDING), 8)

        intersect, _, _, _ = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        agent_screen = w2s(self.agent_pos[0], scale, _SCREEN, _PADDING)
        for i,ray in enumerate(intersect[0]):   # env 0 rays: (num_rays, 2)
            color = (obs["rays"][0, 4:, :].T)[i]
            pygame.draw.line(self.screen, (int(color[0] * 255),
                                           int(color[1] * 255),
                                           int(color[2] * 255)), agent_screen,
                                            w2s(ray, scale, _SCREEN, _PADDING), 1)

        pygame.display.flip()
        self.clock.tick(5)

    def record_frame(self, obs):
        _SCREEN  = 900
        _PADDING = 60

        if not hasattr(self, '_rec_screen'):
            pygame.init()
            self._rec_screen = pygame.Surface((_SCREEN, _SCREEN))

        min_x, max_x, min_y, max_y = self.bounding_box
        world = max(max_x - min_x, max_y - min_y)
        scale = (_SCREEN - 2 * _PADDING) / world
        self._rec_screen.fill((255, 255, 255))

        for start, end in self.walls:
            pygame.draw.line(self._rec_screen, (0, 0, 0),
                             w2s(start, scale, _SCREEN, _PADDING),
                             w2s(end,   scale, _SCREEN, _PADDING), 2)

        pygame.draw.circle(self._rec_screen, (0, 100, 255),
                           w2s(self.agent_pos[0], scale, _SCREEN, _PADDING), 6)
        pygame.draw.circle(self._rec_screen, (0, 255, 0),
                           w2s(self.goal_pos[0], scale, _SCREEN, _PADDING), 8)

        intersect, _, _, _ = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        agent_screen = w2s(self.agent_pos[0], scale, _SCREEN, _PADDING)
        for i, ray in enumerate(intersect[0]):
            color = (obs["rays"][0, 4:, :].T)[i]
            pygame.draw.line(self._rec_screen,
                             (int(color[0]*255), int(color[1]*255), int(color[2]*255)),
                             agent_screen, w2s(ray, scale, _SCREEN, _PADDING), 2)

        frame = np.transpose(pygame.surfarray.array3d(self._rec_screen), (1,0,2))
        return frame

    def record_rollout(self, backbone_type, agent, steps, cfg: DictConfig):
        if backbone_type != "mlp": # Recurrent backbones (lstm, gru)
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
            frames.append(self.record_frame(obs))

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
            frames.append(self.record_frame(obs))

            if done.any():
                obs, _ = self.reset()

        return frames


class NavigationEnvEasy(NavigationEnv):
    """
    Wrapper to NavigationEnv that enhances the observations and reward function to make learning easy.
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

    def reset(self, seed=None, options=None, done: torch.Tensor = None):
        """
        done: bool tensor (num_envs,) — reset only those envs. None resets all.
        Returns (obs, {}) where obs is (num_envs, 4, num_rays) on device.
        """
        mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if done is None else done)

        self.agent_pos[mask]        = self.initial_pos
        self.facing_direction[mask] = torch.pi / 2.0
        self.steps[mask]            = 0
        self.last_speed[mask]       = 0.0
        self.last_turning[mask]     = 0.0
        self._reset_goals(mask)
        self._reset_information(mask)
        self._reset_junction(mask)

        intersections, distances, d_unit, wall_idx = self.ray_cast.scan(self.agent_pos, self.facing_direction)
        self.prev_dist = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        return self.get_observations(intersections, distances, d_unit, wall_idx), {"agent_pos": self.agent_pos}

    def _base_proprio(self) -> list:
        # Proprioceptive: heading on top of the base env's last action
        return [torch.sin(self.facing_direction), torch.cos(self.facing_direction),
                self.last_speed, self.last_turning]

    def compute_reward(self, success, collided=None):

        reward = super().compute_reward(success, collided)
        dist_to_goal = torch.norm(self.agent_pos - self.goal_pos, dim=-1)
        reward += (self.prev_dist - dist_to_goal) / (self.ray_cast.max_range * self.max_steps)

        return reward
