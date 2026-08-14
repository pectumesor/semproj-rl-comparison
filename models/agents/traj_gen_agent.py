import math
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from models.embeddings.grid_cell import GridCellNetwork
from envs import TrajGenEnv
from tqdm import tqdm
from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

WALL_TRESHOLD = 30    # 0.03m (Banino Table 1 d) * 1000 units/m
RAYLEIGH_B = 2.6      # 0.13 m/s (Banino Table 1 sigma^(v)) * 1000 units/m * 0.02 s/step
NORMAL_MU = 0
NORMAL_SIGMA = 6.6    # 330 deg/s (Banino Table 1 sigma^(phi)) * 0.02 s/step
RHO_RH = 0.25         # Banino Table 1: velocity reduction factor near the wall



def rayleigh_sample(b: float, n: int, device=None) -> torch.Tensor:
    """Sample n draws from a Rayleigh(b) distribution via inverse CDF."""
    u = torch.rand(n, device=device)
    return b * torch.sqrt(-2.0 * torch.log(1.0 - u))


class TrajGenAgent(nn.Module):
    def __init__(self, traj_len: int, num_rays: int, obs_dim, cfg: DictConfig, device):
        super().__init__()

        self.N = cfg.place_cell_size
        self.M = cfg.head_dir_cell_size

        self.sigma_c = cfg.sigma_c # Place cell standard deviation parameter
        self.kappa = cfg.kappa # Head direction concentration parameter

        self.grid_cell_network = GridCellNetwork(dropout= cfg.env.dropout,
                                                 place_cell_size=self.N,
                                                 head_dir_cell_size=self.M,
                                                 batch_size=cfg.env.num_envs).to(device)
        
        self.env = TrajGenEnv(cfg=cfg, num_envs=cfg.env.num_envs, obs_dim=obs_dim,
                              num_rays=num_rays, device=device)
        
        self.traj_len = traj_len
        self.nr_iterations = cfg.nr_iterations
        self.block_size = cfg.block_size
        self.device = device

        self.decay, self.no_decay = self.grid_cell_network.select_decay_clip_parameters(
            ["place_cell_head.weight", "place_cell_head.bias",
              "head_dir_head.weight", "head_dir_head.bias"])
        
        self.grad_clip = cfg.clip_tresh

        self.optimizer = torch.optim.RMSprop( [
            {"params": self.decay, "weight_decay": cfg.decay},
            {"params": self.no_decay, "weight_decay": 0.0},], lr=cfg.lr,
              momentum=cfg.momentum)

    def initialize(self):

        min_x, max_x, min_y, max_y = self.env.bounding_box
        x = torch.empty((self.N,), device=self.device).uniform_(min_x + WALL_TRESHOLD, max_x - WALL_TRESHOLD)
        y = torch.empty((self.N,), device=self.device).uniform_(min_y + WALL_TRESHOLD, max_y - WALL_TRESHOLD)

        self.MU_C = torch.stack([x,y], dim=-1) # Place cell centers
        self.MU_H = torch.empty((self.M,), device=self.device).uniform_(- torch.pi, torch.pi) # Head dir cell centers

    
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
        speed_close = random_speed * (1.0 - RHO_RH)

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
        place_cell = torch.softmax(-squared_dists / (2 * self.sigma_c**2), dim=-1)

        squared_thetas = (head_dir[:, None] - self.MU_H)
        head_cell = torch.softmax(self.kappa * torch.cos(squared_thetas), dim=-1)

        return place_cell, head_cell

    def generate_trajectories(self):

        trajectory = []

        obs, _ = self.env.reset()

        distances = obs["rays"].squeeze(1)

        place_cell, head_dir_cell = self.place_and_head_cell_generation(
            pos= obs["proprio"][:, :2], head_dir=obs["proprio"][:, 2]
        )

        trajectory.append(
            {
                "place_cell": place_cell,
                "head_dir_cell":head_dir_cell,
                "linear_speed": torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device),
                "angular_speed": torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
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

            distances = obs["rays"].squeeze(1)

        return trajectory

    def train(self):
        
        self.initialize()
        loss_fn = nn.CrossEntropyLoss()
        self.grid_cell_network.train()
        self.env.compile()

        pbar = tqdm(range(self.nr_iterations), desc="Training Grid Cell Network")
        num_blocks = self.traj_len // self.block_size
        best_loss = float("inf")

        for i in pbar:

            T = self.generate_trajectories()  # traj_len + 1 entries; T[0] is the true reset state

            block_loss_sum = torch.zeros((), device=self.device)

            for b in range(num_blocks):
                start = b * self.block_size

                anchor = T[start]
                c0, h0 = anchor["place_cell"], anchor["head_dir_cell"]
                h = self.grid_cell_network.initialize_lstm(c0, h0)

                block = T[start + 1 : start + 1 + self.block_size]

                linear_speed      = torch.stack([d["linear_speed"]  for d in block], dim=0)  # (B, E)
                angular_speed     = torch.stack([d["angular_speed"] for d in block], dim=0)  # (B, E)
                place_cell_label  = torch.stack([d["place_cell"]    for d in block], dim=0)  # (B, E, N)
                head_dir_label    = torch.stack([d["head_dir_cell"] for d in block], dim=0)  # (B, E, M)

                x = torch.stack([linear_speed, torch.sin(angular_speed),
                                  torch.cos(angular_speed)], dim=-1).to(self.device)          # (B, E, 3)

                place_cell_pred, head_dir_pred, _ = self.grid_cell_network(x, h)

                loss = loss_fn(place_cell_pred.reshape(-1, place_cell_pred.shape[-1]),
                                place_cell_label.reshape(-1, place_cell_label.shape[-1])) \
                     + loss_fn(head_dir_pred.reshape(-1, head_dir_pred.shape[-1]),
                                head_dir_label.reshape(-1, head_dir_label.shape[-1]))

                # One gradient step per block. LSTM state is discarded across blocks —
                # each block re-anchors on the true position instead of carrying state over.
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.decay, clip_value=self.grad_clip)
                self.optimizer.step()

                block_loss_sum += loss.detach()

            curr_loss = (block_loss_sum / num_blocks).item()
            pbar.set_description(f"Loss at iteration {i+1}: {curr_loss:.4f}")

            if (i+1) % 1000 == 0 and curr_loss < best_loss:
                best_loss = curr_loss
                path = ROOT_DIR / "logs" / "grid_cell_test" / f"loss_{best_loss:.4f}"
                self.save_model(path)



            #print(f"Loss at iteration {i + 1}: {loss.item():4f}")

    def save_model(self, path):
    
            checkpoint = {
                "grid_network":self.grid_cell_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
         
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)
    
    def load_model(self, path, device):
    
        checkpoint = torch.load(path, map_location=device)
    
        self.grid_cell_network.load_state_dict(checkpoint["grid_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    







            







