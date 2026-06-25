import torch
import numpy as np
import json


def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2-D cross product (scalar) on the last axis: a[...,0]*b[...,1] - a[...,1]*b[...,0]."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


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

    def __init__(self, cfg, wall_starts: torch.Tensor, wall_ends: torch.Tensor, num_rays: int):
        self.max_range = cfg.env.max_range
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

    def scan(self, positions: torch.Tensor, facing_directions: torch.Tensor):
        """
        positions:         (E, 2)
        facing_directions: (E,)
        Returns:
            intersections: (E, R, 2)   — position of closest wall hit per ray (origin for misses)
            distances:     (E, R)      — inf when no wall hit
        """
        # Ray directions
        angles = facing_directions[:, None] + self._ray_offsets[None, :]   # (E, R)
        d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * self.max_range
        # d: (E, R, 2)

        r = self._r                                                          # (W, 2)
        e = self.wall_starts[None, :, :] - positions[:, None, :]           # (E, W, 2)

        # e_cross_r[E, W] — same for every ray of an env
        e_cross_r = _cross2d(e, r[None, :, :])[:, None, :]                 # (E, 1, W)

        d_exp = d[:, :, None, :]        # (E, R, 1, 2)
        r_exp = r[None, None, :, :]     # (1, 1, W, 2)
        e_exp = e[:, None, :, :]        # (E, 1, W, 2)

        d_cross_r = _cross2d(d_exp, r_exp) + 1e-8   # (E, R, W)
        e_cross_d = _cross2d(e_exp, d_exp)           # (E, R, W)

        t = e_cross_r / d_cross_r   # (E, R, W)
        s = e_cross_d / d_cross_r   # (E, R, W)

        hit    = (t >= 0) & (t <= 1.0) & (s >= 0) & (s <= 1.0)
        t_hits = torch.where(hit, t, torch.full_like(t, float('inf')))
        min_t, _ = t_hits.min(dim=-1)                                       # (E, R)

        distances = min_t * self.max_range

        # Clamp inf to 0 before multiply to avoid nan (no-hit rays get origin; masked out later)
        min_t_safe = min_t.masked_fill(torch.isinf(min_t), 0.0)
        intersections = positions[:, None, :] + min_t_safe[:, :, None] * d # (E, R, 2)

        return intersections, distances


# ---------------------------------------------------------------------------
# Utility functions (numpy — used only at startup to load room geometry)
# ---------------------------------------------------------------------------

def walls_json_to_numpy(json_path: str):
    walls = []
    with open(json_path) as f:
        for edge in json.load(f)["edges"]:
            walls.append(([edge["from"]["x"], edge["from"]["y"]],
                          [edge["to"]["x"],   edge["to"]["y"]]))
    return walls


def compute_starts_and_ends(walls):
    wall_starts = np.array([p for p, _ in walls], dtype=np.float32)
    wall_ends   = np.array([q for _, q in walls], dtype=np.float32)
    return wall_starts, wall_ends


def compute_num_rays(fov, ray_density):
    num_rays = int(fov * ray_density)
    if num_rays % 2 == 0:
        num_rays += 1
    return num_rays
