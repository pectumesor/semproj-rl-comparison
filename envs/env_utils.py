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

    def intersect(self, positions: torch.Tensor, d: torch.Tensor):
        """
        Shared ray-wall intersection kernel.

        positions: (E, 2)
        d:         (E, R, 2)  ray vectors (not necessarily unit; length = max travel)

        Returns:
            min_t:  (E, R)  in [0,1] fraction of |d| to closest wall; inf = no hit
        """
        r = self._r                                              # (W, 2)
        e = self.wall_starts[None, :, :] - positions[:, None, :]  # (E, W, 2)

        e_cross_r = _cross2d(e, r[None, :, :])[:, None, :]     # (E, 1, W)

        d_exp = d[:, :, None, :]        # (E, R, 1, 2)
        r_exp = r[None, None, :, :]     # (1, 1, W, 2)
        e_exp = e[:, None, :, :]        # (E, 1, W, 2)

        d_cross_r = _cross2d(d_exp, r_exp) + 1e-8  # (E, R, W)
        e_cross_d = _cross2d(e_exp, d_exp)          # (E, R, W)

        t = e_cross_r / d_cross_r  # (E, R, W)
        s = e_cross_d / d_cross_r  # (E, R, W)

        hit    = (t >= 0) & (t <= 1.0) & (s >= 0) & (s <= 1.0)
        t_hits = torch.where(hit, t, torch.full_like(t, float('inf')))
        min_t, _ = t_hits.min(dim=-1)  # (E, R)
        return min_t

    def scan(self, positions: torch.Tensor, facing_directions: torch.Tensor):
        """
        positions:         (E, 2)
        facing_directions: (E,)
        Returns:
            intersections: (E, R, 2)   — global hit position; ray endpoint at max_range for misses
            distances:     (E, R)      — inf when no wall hit
            d_unit:        (E, R, 2)   — unit ray directions (reuse in get_observations)
        """
        angles = facing_directions[:, None] + self._ray_offsets[None, :]        # (E, R)
        d_unit = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)    # (E, R, 2)
        d      = d_unit * self.max_range

        min_t = self.intersect(positions, d)                                     # (E, R)
        distances = min_t * self.max_range

        min_t_safe   = min_t.masked_fill(torch.isinf(min_t), 1.0)
        intersections = positions[:, None, :] + min_t_safe[:, :, None] * d      # (E, R, 2)

        return intersections, distances, d_unit



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

def w2s(pos, scale, screen_size, padding):
            """World (x, y) → pygame pixel (px, py) with Y-flip and padding."""
            return (int(float(pos[0]) * scale) + padding,
                    int(screen_size - padding - float(pos[1]) * scale))
