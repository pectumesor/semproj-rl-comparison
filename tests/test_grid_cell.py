"""
Compare my own PPO implementation with Stable Baselines

"""
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models import TrajGenAgent

#Env
from envs import compute_num_rays

from wandb.integration.sb3 import WandbCallback


import torch
import torch.nn as nn
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="train_grid_cell", version_base=None)
def main(cfg: DictConfig):

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    #wandb.login()

    #with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True),
    #                 sync_tensorboard=True):
        
        T = 700  # 15s / Δt=0.02s (Banino Table 1) ≈ 750, rounded to a multiple of block_size=100

        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        agent = TrajGenAgent(traj_len=T, cfg=cfg, num_rays=num_rays, obs_dim=tuple(ray_dim), device=device)

        agent.train()

if __name__ == "__main__":
    main()
