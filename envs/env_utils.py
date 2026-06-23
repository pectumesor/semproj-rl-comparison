import numpy as np
import json

class RayCast:
    def __init__(self, cfg, wall_starts, wall_ends, num_rays):

        self.max_range = cfg.env.max_range
        self.ray_density = cfg.env.ray_density
        self.fov = cfg.env.fov # In degrees
        self.wall_ends = wall_ends
        self.wall_starts = wall_starts
        self.num_rays = num_rays
    
    def cast_rays(self, facing_direction):
        
        half_fov = np.deg2rad(self.fov // 2)
        angles = np.linspace(facing_direction - half_fov, 
                           facing_direction + half_fov, 
                           self.num_rays)
        
        return np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.max_range
    
    def scan(self, position, facing_direction):

        """
        
        Rays: o + t * d

        Walls: p + s* (q-p), for walls = (p,q)

        For e = p - o, and r = q - p

        Need to solve:

            t = e x r / (d x r)
            s = e x d / (d x r)

            where "x" is the cross product
        
        t >= 0: Hit is in front of the ray origin 
        0 <= s <= 1: Hit lands within the wall segment
        
        """

        d = self.cast_rays(facing_direction) 
        o = position

        r = self.wall_ends - self.wall_starts
        e = self.wall_starts - o

        d_ = d[:, None, :]
        e_ = e[None, :, :]
        r_ = r[None, :, :]

        e_cross_r = np.cross(e_,r_)
        e_cross_d = np.cross(e_,d_)
        d_cross_r = np.cross(d_,r_) + 1e-8 # Avoid division by zero on parallel segments

        t = e_cross_r / d_cross_r
        s = e_cross_d / d_cross_r

        hit = (t >= 0) & (t <= 1.0) & (s >=0 ) & (s <= 1.0)

        t_hits = np.where(hit, t, np.inf)
        min_t = t_hits.min(axis=1)
        distances = min_t * self.max_range

        # Compute intersection coordinates
        intersections = o + min_t[:, np.newaxis]

        return intersections, distances
    


def walls_json_to_numpy(json_path: str) -> np.ndarray:

    walls = []

    with open(json_path) as f:
        walls_dict = json.load(f)
        for edge in walls_dict["edges"]:
            p = [edge["from"]["x"], edge["from"]["y"]]
            q = [edge["to"]["x"], edge["to"]["y"]]
            walls.append((p, q))

    return walls
             

def compute_starts_and_ends(walls):

    wall_starts = np.array([p for p,_ in walls])
    wall_ends = np.array([q for _, q in walls])

    return wall_starts, wall_ends

def compute_num_rays(fov, ray_density):

    num_rays = int(fov * ray_density)

    """
        Keep an uneven number of rays,
        so we have one ray at the facing direction and the others
        evenly spread on each half of the fov
    """
    if num_rays % 2 == 0:
            num_rays += 1
    
    return num_rays




    












