import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

class RayCast:
    def __init__(self, cfg):

        self.max_range = cfg['max_range']
        self.ray_density = cfg['ray_density']
        self.fov = cfg['field_of_view'] # In degrees
        self.wall_ends = None
        self.wall_starts = None
    
    def cast_rays(self, facing_direction):

        num_rays = int(self.fov * self.ray_density)

        """
        Keep an uneven number of rays,
        so we have one ray at the facing direction and the others
        evenly spread on each half of the fov
        """
        if num_rays % 2 == 0:
            num_rays += 1

        
        half_fov = np.deg2rad(self.fov // 2)
        angles = np.linspace(facing_direction - half_fov, 
                           facing_direction + half_fov, 
                           num_rays)
        
        return np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.max_range
    
    def scan(self, position, walls, facing_direction):

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
        o = position.cpu().detach().numpy()

        if self.wall_starts is None or self.wall_ends is None:
            self.wall_starts = np.array([p for p,_ in walls])
            self.wall_ends = np.array([q for _, q in walls])

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
    
    












