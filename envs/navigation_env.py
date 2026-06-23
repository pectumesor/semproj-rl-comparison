import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from .env_utils import RayCast, walls_json_to_numpy, compute_starts_and_ends 


class NavigationEnv(gym.Env):

    def __init__(self, cfg, room_path: str, agent: nn.Module):
        metadata = {"render_modes": ["human"], "render_fps": 10}

        super().__init__()

        self.act_dim = cfg['action_dim']
        self.obs_dim = cfg['obs_dim']
        self.max_speed = cfg['max_vel']
        self.fov = cfg['field_of_view']
        self.max_steps = cfg['max_steps']
        self.initial_pos = cfg['init_pos']
        self.goal_pos = cfg['goal_pos']

        """
        
        -- Observation Space --

        A matrix of dimension (num_classes + 1) x num_rays

        Each column is a one-hot encodded vector of size: num_classes.
        The last entry of a column is a normalized distance of how far the ray was cast
        
        """

        self.observation_space = gym.Box(
            low=0.0,
            high=1.0,
            dtype = np.float32,
            shape = (self.obs_dim,)
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

        self.action_space = gym.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.walls = walls_json_to_numpy(room_path)
        self.agent = agent

        # Agent variables
        wall_starts, wall_ends = compute_starts_and_ends(self.walls)
        self.ray_cast = RayCast(cfg, wall_starts, wall_ends)
        self.facing_direction = np.pi / 2
        self.agent_pos = self.initial_pos
       
        self.steps = 0


    def reset(self, seed=None, options=None):
        # TODO: Implement
        super().reset(seed=seed)
        
        self.facing_direction = np.pi / 2
        self.agent_pos = self.initial_pos

        intersections, distances = self.ray_cast.scan(
            self.agent_pos, self.facing_direction)
        
        obs = self.get_observations(intersections, distances)

        return obs, {}

    def get_observations(self, intersections, distances):

        list = []
        max_range = self.ray_cast.max_range


        for i in range(len(distances)):

            if distances[i] == np.inf:
                list.append(np.array([1,0,0,max_range]))
            elif distances[i] < np.inf and (
                np.linalg.norm(intersections[i] - self.goal_pos) <= 1e-4
            ):
                list.append(np.array([0,1,0,distances[i]]))
            else:
                list.append(np.array([0,0,1,distances[i]]))
        

        obs = np.stack(list, axis=-1)

        return obs
    
    def compute_rewards(self, done):

        reward = -(1.0 / self.max_steps)

        if done:
            reward += 1
        
        return reward
                
    def step(self, action):

        done = False
        truncated = False

        self.steps += 1

        if self.steps >= self.max_steps:
            self.steps = 0
            truncated = True

        turning_angle = action[0] * self.fƒov
        velocity = action[1] * self.max_speed

        self.agent_pos += velocity
        self.facing_direction += turning_angle

        intersections, distances = self.ray_cast.scan(self.agent_pos, self.facing_direction)

        obs = self.get_observations(intersections, distances)

        if np.linalg.norm(self.agent_pos - self.goal_pos) <= 1e-4:
            done = True
        
        reward = self.compute_rewards(done)

        info = {
            "Facing Direction": self.facing_direction,
            "Current Position": self.agent_pos,
            "Last Turning Angle": turning_angle,
            "Last Velocity": velocity 
        }

        return obs, reward, done, truncated, info







