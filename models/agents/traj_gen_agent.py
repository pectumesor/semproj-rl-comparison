import math
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from models import GridCellNetwork
from envs import TrajGenEnv

WALL_TRESHOLD = 2

RAYLEIGH_B = 13
NORMAL_MU = 0
NORMAL_SIGMA = 330


def rayleigh_sample(b: float, n: int, device=None) -> torch.Tensor:
    """Sample n draws from a Rayleigh(b) distribution via inverse CDF."""
    u = torch.rand(n, device=device)
    return b * torch.sqrt(-2.0 * torch.log(1.0 - u))


class TrajGenAgent(nn.Module):
    def __init__(self, traj_len: int, cfg: DictConfig, device):
        super().__init__()

        self.N = cfg.model.place_cell_size
        self.M = cfg.model.head_dir_cell_size

        self.sigma_c = cfg.model.sigma_c # Place cell standard deviation parameter
        self.kappa = cfg.model.kappa # Head direction concentration parameter

        self.grid_cell_network = GridCellNetwork(dropout= cfg.env.dropout,
                                                 place_cell_size=self.N,
                                                 head_dir_cell_size=self.M,
                                                 batch_size=cfg.env.num_envs).to(device)
        
        self.env = TrajGenEnv(cfg=cfg, num_envs=cfg.env.num_rays, obs_dim=cfg.env.obs_dim,
                              num_rays=cfg.env.num_envs, device=device)
        
        self.traj_len = traj_len
        self.nr_iterations = cfg.model.iterations
        self.device = device

        self.optimizer = torch.optim.AdamW(self.grid_cell_network.parameters(), lr=cfg.model.lr)

    def initialize(self):

        min_x, max_x, min_y, max_y = self.env.bounding_box
        x = torch.empty((self.N,)).uniform_(min_x + WALL_TRESHOLD, max_x - WALL_TRESHOLD)
        y = torch.empty((self.N,)).uniform_(min_y + WALL_TRESHOLD, max_y - WALL_TRESHOLD)

        self.MU_C = torch.stack([x,y], dim=-1) # Place cell centers
        self.MU_H = torch.empty((self.M,)).uniform_(- torch.pi, torch.pi) # Head dir cell centers

    
    def get_action(self, distances: torch.Tensor) -> torch.Tensor:
        """
        distances: (num_envs, num_rays) ray-cast distances for the current heading.
        Returns action: (num_envs, 2) normalized to [-1, 1] for TrajGenEnv.step.
        """
        E = self.env.num_envs

        half_fov  = self.env._half_fov_rad
        num_rays  = self.env.num_rays
        max_speed = self.env.max_speed

        d_wall, min_ray = torch.min(distances, dim=-1)  # (E,), (E,)

        ray_angles = torch.linspace(
            -half_fov, half_fov, num_rays,
            dtype=torch.float32, device=self.device,
        )
        a_wall = ray_angles[min_ray]  # (E,) angle to nearest wall, relative to heading

        mu    = math.radians(NORMAL_MU)
        sigma = math.radians(NORMAL_SIGMA)

        random_turn  = mu + sigma * torch.randn(E, device=self.device)        # (E,) radians
        random_speed = rayleigh_sample(RAYLEIGH_B, E, device=self.device)     # (E,)

        close = (d_wall < WALL_TRESHOLD) & (a_wall.abs() < torch.pi / 2.0)    # (E,) bool

        angle_close = torch.sign(a_wall) * (torch.pi / 2.0 - a_wall.abs()) + random_turn
        speed_close = random_speed - 0.5 * (random_speed - 5.0)

        angle = torch.where(close, angle_close, random_turn)
        speed = torch.where(close, speed_close, random_speed)

        angle = torch.clamp(angle, -half_fov, half_fov)
        speed = torch.clamp(speed, 0.0, max_speed)

        return torch.stack([angle / half_fov, speed / max_speed], dim=-1)  # (E, 2)

    def place_and_head_cell_generation(self, pos:torch.Tensor, head_dir:torch.Tensor):

        #Pos shape: (E, 2) -> Place cell shape: (E, N)
        #Head_dir Shape: (E,) -> Head dir cell shape: (E, M) 

        # MU_C shape: (N,2), MU_H Shape: (M,)

        squared_dists = torch.sum((pos[:, None, :] - self.MU_C[None, :, :]) ** 2, dim=-1) # Second Norm
        place_cell = torch.softmax(-squared_dists / (2 * self.sigma_c)**2, dim=-1)

        squared_thetas = (head_dir[:, None] - self.MU_H)
        head_cell = torch.softmax(self.kappa * torch.cos(squared_thetas), dim=-1)

        return place_cell, head_cell

    def generate_trajectories(self):

        trajectory = []

        obs = self.env.reset()

        distances = obs["rays"]

        place_cell, head_dir_cell = self.place_and_head_cell_generation(
            pos= obs["proprio"][:, :2], head_dir=obs["proprio"][:, 2]
        )

        trajectory.append(
            {
                "place_cell": place_cell,
                "head_dir_cell":head_dir_cell,
                "linear_speed": torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device),
                "angular_peed": torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
            }
        )

        for _ in range(self.traj_len):

            action = self.get_action(distances)
            obs, speeds = self.env.step(action)

            place_cell, head_dir_cell = self.place_and_head_cell_generation(
            pos= obs["proprio"][:, :2], head_dir=obs["proprio"][:, 2] )

            trajectory.append({
                "place_cell": place_cell,
                "head_dir_cell": head_dir_cell,
                "linear_speed": speeds["last_speed"],
                "angular_speed": speeds["last_turning"]
            })

            distances = obs["rays"]

        return trajectory

    def train(self):
        
        self.initialize()
        loss_fn = nn.CrossEntropyLoss()
        self.grid_cell_network.train()

        total_loss = 0
        total_size = 0

        for i in range(self.nr_iterations):

            T = self.generate_trajectories()

            data = T.pop(0)
            c0, h0 = data["place_cell"], data["head_dir_cell"]
            self.grid_cell_network.initial_lstm_state(c0, h0)

            for data in T:

                place_cell_label = data["place_cell"]
                head_dir_label = data["head_dir_cell"]
                linear_speed = data["linear_speed"]
                angular_speed = data["angular_speed"]

                total_size += place_cell_label.shape(0)

                x = torch.cat([linear_speed, torch.sin(angular_speed),
                               torch.cos(angular_speed)], dim=-1).to(self.device)       

                place_cell_pred, head_dir_pred = self.grid_cell_network(x)

                loss = loss_fn(place_cell_pred,
                                place_cell_label) + loss_fn(head_dir_pred, head_dir_label)

                total_loss += loss.item()

                # Backpropragation
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()

            
            total_loss /= total_size

            print(f"Loss at iteration {i + 1}: {total_loss:4f}")









            







