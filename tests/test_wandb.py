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
# Architecture pieces
from models import (MLPObservationEmbeddings,
                    MLPBackbone, GuassianPolicyHead,
                    ValueNet, SquashedGaussianPolicyHead, DoubleQNet)
# Agents
from models import BaseAgent
from algorithms import RolloutBuffer, MLPPPO, MLPSAC, ReplayBuffer

#Env
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3, MyBackbone
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="test_wandb", version_base=None)
def main(cfg: DictConfig):

    wandb.login()

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True)):

        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)

        # Derive shapes from the env directly so they never go out of sync with navigation_env.py
        _probe = NavigationEnvEasy(cfg, None, num_rays, (1, num_rays), num_envs=1)
        _obs, _ = _probe.reset()
        ray_dim     = tuple(_obs["rays"].shape[1:])    # (C, R)
        proprio_dim = _obs["proprio"].shape[-1]         # 2
        del _probe

        observation_model = MLPObservationEmbeddings(
            input_dim=int(np.prod(ray_dim)) + proprio_dim,
            hidden_sizes=cfg.model.obs_embed_hidden_sizes,
            feature_dim=cfg.model.obs_embed_hidden_sizes[-1]
        )

        backbone_model = MLPBackbone(
            input_dim=cfg.model.obs_embed_hidden_sizes[-1],
            hidden_sizes=cfg.model.backbone_hidden_sizes,
            output_dim=cfg.model.backbone_hidden_sizes[-1]
        )

        actor = GuassianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                    actions_dim=cfg.env.act_dim,
                                    hidden_sizes=cfg.model.policy_hidden_sizes)
            
        critic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                        hidden_sizes=cfg.model.value_hidden_sizes)

        agent = BaseAgent(obs_embed_model=observation_model,
                        backbone_model=backbone_model,
                        actor=actor, critic=critic).to(device)

        env      = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1,               device=device).compile()
        buffer = RolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device,cfg=cfg)
        algorithm = MLPPPO(buffer=buffer,device=device,env=env,eval_env=eval_env, agent=agent, cfg=cfg)
    
        
        log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}"
        run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
        run_dir = log_dir / run_name
        
        algorithm.train(run_dir=run_dir)

if __name__ == "__main__":
    main()
