import torch
import numpy as np

from utils.geometry import cross2d


class RayCast:
    """
    Fully vectorized ray-wall intersection over (num_envs, num_rays) simultaneously.
    All tensors live on device; no numpy after construction.

    Ray:  P(t) = o + t*d,  d scaled by max_range
    Wall: Q(s) = p + s*r,  r = q - p

        t = (e × r) / (d × r)
        s = (e × d) / (d × r)
        e = p - o
    """

    def __init__(self, cfg, wall_starts: torch.Tensor, wall_ends: torch.Tensor, num_rays: int, max_range: int):
        self.max_range = max_range
        self.fov       = cfg.env.fov
        self.num_rays  = num_rays

        self.wall_starts = wall_starts          # (W, 2)
        self.wall_ends   = wall_ends            # (W, 2)
        self._r          = wall_ends - wall_starts  # (W, 2) precomputed wall directions

        half_fov = float(np.deg2rad(self.fov / 2.0))
        self._ray_offsets = torch.linspace(
            -half_fov, half_fov, num_rays,
            dtype=torch.float32, device=wall_starts.device,
        )

    def to(self, device):
        self.wall_starts  = self.wall_starts.to(device)
        self.wall_ends    = self.wall_ends.to(device)
        self._r           = self._r.to(device)
        self._ray_offsets = self._ray_offsets.to(device)
        return self

    def intersect(self, positions: torch.Tensor, d: torch.Tensor, return_index: bool = False):
        """
        Shared ray-wall intersection kernel.

        positions: (E, 2)
        d:         (E, R, 2)  ray vectors (not necessarily unit; length = max travel)

        Returns:
            min_t:    (E, R)  in [0,1] fraction of |d| to closest wall; inf = no hit
            wall_idx: (E, R)  index of the closest wall, -1 = no hit (only if return_index)
        """
        r = self._r                                              # (W, 2)
        e = self.wall_starts[None, :, :] - positions[:, None, :]  # (E, W, 2)

        e_cross_r = cross2d(e, r[None, :, :])[:, None, :]     # (E, 1, W)

        d_exp = d[:, :, None, :]        # (E, R, 1, 2)
        r_exp = r[None, None, :, :]     # (1, 1, W, 2)
        e_exp = e[:, None, :, :]        # (E, 1, W, 2)

        d_cross_r = cross2d(d_exp, r_exp) + 1e-8  # (E, R, W)
        e_cross_d = cross2d(e_exp, d_exp)          # (E, R, W)

        t = e_cross_r / d_cross_r  # (E, R, W)
        s = e_cross_d / d_cross_r  # (E, R, W)

        hit    = (t >= 0) & (t <= 1.0) & (s >= 0) & (s <= 1.0)
        t_hits = torch.where(hit, t, torch.full_like(t, float('inf')))
        min_t, wall_idx = t_hits.min(dim=-1)  # (E, R)
        if return_index:
            # Which wall was hit, so walls can carry their own colour (per_wall mode and cue walls)
            return min_t, wall_idx.masked_fill(torch.isinf(min_t), -1)
        return min_t

    def scan(self, positions: torch.Tensor, facing_directions: torch.Tensor):
        """
        positions:         (E, 2)
        facing_directions: (E,)
        Returns:
            intersections: (E, R, 2)   — global hit position; ray endpoint at max_range for misses
            distances:     (E, R)      — inf when no wall hit
            d_unit:        (E, R, 2)   — unit ray directions (reuse in get_observations)
            wall_idx:      (E, R)      — index of the hit wall, -1 when no wall hit
        """
        angles = facing_directions[:, None] + self._ray_offsets[None, :]        # (E, R)
        d_unit = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)    # (E, R, 2)
        d      = d_unit * self.max_range

        min_t, wall_idx = self.intersect(positions, d, return_index=True)        # (E, R)
        distances = min_t * self.max_range

        min_t_safe   = min_t.masked_fill(torch.isinf(min_t), 1.0)
        intersections = positions[:, None, :] + min_t_safe[:, :, None] * d      # (E, R, 2)

        return intersections, distances, d_unit, wall_idx

class PerlinColor:
    """
    Continuous, pointwise-queryable Perlin color field for state disambiguation.

    Produces 3 decorrelated noise channels → an RGB color that is spatially
    coherent (nearby points → similar color) but decorrelated across the map
    (far-apart points → independent color). Same field every call given a fixed
    seed, so colors carry stable spatial information the policy can learn.

    Query at any continuous (x, y); shapes broadcast, so (E, R) hit coords in,
    (E, R, 3) RGB out.
    """

    def __init__(self, seed=0, scale=0.15, octaves=4,
                 persistence=0.5, lacunarity=2.0, device="cpu"):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.device = device

        # 3 decorrelated permutation tables (different seeds, large stride apart)
        self.perms = []
        for i in range(3):
            g = torch.Generator(device="cpu").manual_seed(seed + i * 7919)
            p = torch.randperm(256, generator=g)
            p = torch.cat([p, p]).to(device)          # 512, avoids wrap masking
            self.perms.append(p)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)    # quintic smoothstep

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def _grad(h, x, y):
        # 8 gradient directions from low 3 bits of the hash
        h = h & 7
        u = torch.where(h < 4, x, y)
        v = torch.where(h < 4, y, x)
        return (torch.where((h & 1) == 0, u, -u) +
                torch.where((h & 2) == 0, v, -v))

    def _noise(self, perm, x, y):
        # x, y: arbitrary shape, float. Returns same shape in ~[-1, 1].
        xi = torch.floor(x).long() & 255
        yi = torch.floor(y).long() & 255
        xf = x - torch.floor(x)
        yf = y - torch.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        # hash the 4 lattice corners
        aa = perm[perm[xi]     + yi]
        ab = perm[perm[xi]     + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]

        x1 = self._lerp(self._grad(aa, xf,     yf),
                        self._grad(ba, xf - 1, yf),     u)
        x2 = self._lerp(self._grad(ab, xf,     yf - 1),
                        self._grad(bb, xf - 1, yf - 1), u)
        return self._lerp(x1, x2, v)

    def _fbm(self, perm, x, y):
        freqs = self.lacunarity ** torch.arange(self.octaves, dtype=torch.float32, device=x.device)
        amps  = self.persistence ** torch.arange(self.octaves, dtype=torch.float32, device=x.device)
        s = (-1,) + (1,) * x.dim()
        vals = self._noise(perm, x.unsqueeze(0) * freqs.view(s),
                                 y.unsqueeze(0) * freqs.view(s))   # (O, *shape)
        return (vals * amps.view(s)).sum(0) / amps.sum()            # ~[-1, 1]

    def __call__(self, x, y):
        """
        x, y: float tensors of identical shape, e.g. (E, R) hit coordinates.
        Returns RGB in [0, 1] with a trailing channel dim, e.g. (E, R, 3).
        """
        x = (x * self.scale).to(self.device)
        y = (y * self.scale).to(self.device)
        chans = [self._fbm(p, x, y) for p in self.perms]   # 3 × (E, R)
        rgb = torch.stack(chans, dim=-1)                   # (E, R, 3)
        return (rgb + 1.0) * 0.5                            # → [0, 1]

class GridCode:
    """
    Hand-crafted grid cell code of a 2D position (Solstad, Moser & Einevoll 2006).

    Each cell fires as the sum of three plane waves 60° apart:
        g(x) = (sum_k cos(k_k · (x - c)) + 1.5) / 4.5   in [0, 1]
    with |k_k| = 4π / (√3 λ). Every module has its own spacing λ_m = min_spacing * spacing_ratio^m
    and a random orientation, and holds `phases` cells with random offsets c inside the unit cell.
    No training needed: a fixed, layout-agnostic alternative to the learned grid cells of Banino et al.

    Query with (..., 2) positions, returns (..., modules * phases).
    """

    def __init__(self, modules=4, phases=16, min_spacing=60.0, spacing_ratio=1.42, seed=0, device="cpu"):
        g = torch.Generator(device="cpu").manual_seed(seed)

        spacings     = min_spacing * spacing_ratio ** torch.arange(modules, dtype=torch.float32)     # (M,)
        orientations = torch.rand(modules, generator=g) * (torch.pi / 3)                             # (M,) hex symmetry repeats every 60°
        wave_angles  = orientations[:, None] + torch.arange(3) * (torch.pi / 3)                      # (M, 3)
        wave_norm    = 4 * torch.pi / (np.sqrt(3) * spacings)                                        # (M,)
        k = wave_norm[:, None, None] * torch.stack([torch.cos(wave_angles), torch.sin(wave_angles)], dim=-1)  # (M, 3, 2)

        # One random phase per cell, uniform over a spacing x spacing patch (covers the unit cell)
        offsets = torch.rand(modules, phases, 2, generator=g) * spacings[:, None, None]               # (M, P, 2)

        self.k = k.to(device)
        self.offsets = offsets.to(device)
        self.dim = modules * phases

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        diff = pos[..., None, None, :] - self.offsets                              # (..., M, P, 2)
        phase = torch.einsum("...mpd,mkd->...mpk", diff, self.k)                   # (..., M, P, 3)
        g = (torch.cos(phase).sum(dim=-1) + 1.5) / 4.5                             # (..., M, P) in [0, 1]
        return g.flatten(-2)

class RandomFourierFeatures:
    """
    Random Fourier features of a 2D position (Tancik et al. 2020), the control for GridCode:
    same output size and frequency range, but random directions instead of the 60° hexagonal structure.

    Query with (..., 2) positions, returns (..., dim) in [0, 1].
    """

    def __init__(self, dim=64, min_spacing=60.0, max_spacing=60.0 * 1.42 ** 3, seed=0, device="cpu"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        num_freqs = dim // 2

        # Wavelengths log-uniform over the same range as the grid modules, random directions
        log_spacing = torch.rand(num_freqs, generator=g) * (np.log(max_spacing) - np.log(min_spacing)) + np.log(min_spacing)
        angles = torch.rand(num_freqs, generator=g) * 2 * torch.pi
        freqs = (2 * torch.pi / torch.exp(log_spacing))[:, None] * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        self.freqs = freqs.to(device)                                              # (F, 2)
        self.shift = (torch.rand(num_freqs, generator=g) * 2 * torch.pi).to(device)
        self.dim = 2 * num_freqs

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        phase = pos @ self.freqs.T + self.shift                                    # (..., F)
        return (torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1) + 1.0) * 0.5

# Discrete action sets as (turning, speed) in the continuous action's normalized units.
# The agent picks an index, the env still receives a continuous action.
DISCRETE_ACTION_TABLES = {
    3: [[-0.5, 1.0], [0.0, 1.0], [0.5, 1.0]],
    5: [[-1.0, 0.5], [-0.5, 1.0], [0.0, 1.0], [0.5, 1.0], [1.0, 0.5]],
}

GOAL_INFO_DIM = 5   # (distance, cos bearing, sin bearing, seen flag, age)

def information_dims(cfg):
    """
    Sizes of the information slots appended to the proprio vector: (goal, position, goal grid).
    With env.reserve_info_slots every slot is allocated (zeros when unused), so conditions that
    differ only in what the agent is told share the same input size.
    """
    info = cfg.env.information
    grid_dim = info.grid.modules * info.grid.phases
    reserve = bool(info.reserve_info_slots)

    goal_dim = GOAL_INFO_DIM if (reserve or info.goal_info != "none") else 0
    if reserve:
        position_dim = grid_dim
    elif info.position_info == "none":
        position_dim = 0
    elif info.position_info == "xy":
        position_dim = 2
    else:
        position_dim = grid_dim
    goal_grid_dim = grid_dim if (reserve or info.goal_grid) else 0

    return goal_dim, position_dim, goal_grid_dim

def compute_proprio_dim(cfg):
    """Base proprio (last speed, last turning, ...) plus the information slots."""
    return cfg.env.proprio_dim + sum(information_dims(cfg))

class LoopLabeler:
    def __init__(self, num_envs, max_len, eta1, eta2, env, device):
        self.eta1 = eta1
        self.eta2 = eta2
        self.num_envs = num_envs
        self.max_len = max_len
        self.env = env
        self.device = device


        self.traj = torch.zeros((num_envs,max_len, 2 ), dtype=torch.float, device=device)
        # Keeps track of the current trajectory length for every environment
        self.len = torch.zeros(num_envs, dtype=torch.long, device=device) 
        self.steps = torch.arange(max_len, device=device)
        self.env_idxs = torch.arange(num_envs, device=device)
    
    def reset_traj(self, episode_starts):
      self.len[episode_starts] = 0

    def step(self):
        pos = self.env.agent_pos # (E, 2)

        valid = self.steps[None, :] < self.len[:, None]                  # (E, T)
        dist = torch.linalg.norm(self.traj - pos[:, None, :], dim=-1)    # (E, T)

        close = (dist < self.eta1) & valid
        far = (dist > self.eta2) & valid
        close_seen = torch.cummax(close.int(), dim=1).values.bool()
        labels = (close_seen & far).any(dim=1).float()

        self.traj[self.env_idxs, self.len.clamp(max=self.max_len - 1)] = pos
        self.len = (self.len + 1).clamp(max=self.max_len)
        return labels
