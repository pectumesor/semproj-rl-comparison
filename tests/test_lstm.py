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
                    ValueNet)
# Agents
from models import PPOAgent, RecurrentPPOAgent
from algorithms import RolloutBuffer, MLPPPO, RecuurentPPO
from utils import create_ppo_agent, create_sb3_ppo_agent, save_video

#Env
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3, MyBackbone
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback


import torch
import torch.nn as nn
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="test_wandb", version_base=None)
def main(cfg: DictConfig):

    wandb.login()

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True),
                     sync_tensorboard=True):

        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        agent = create_ppo_agent("MLP","LSTM", ray_dim, cfg).to(device)

        env   = NavigationEnvEasy(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=ray_dim,
                                           num_envs=cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnvEasy(cfg=cfg, agent=agent, num_rays=num_rays, 
                                     obs_dim=ray_dim, num_envs=1, device=device).compile()
        buffer = RolloutBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device,cfg=cfg)
        algorithm = MLPPPO(buffer=buffer,device=device,env=env,eval_env=eval_env, agent=agent, cfg=cfg)

        vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg=cfg,
                                                         num_rays=num_rays,
                                                          ray_dim=ray_dim,
                                                           proprio_dim=4), n_envs=cfg.env.num_envs)

        sb3_agent = create_sb3_ppo_agent("LSTM", cfg, ray_dim, vec_env)

        log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}"
        run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
        run_dir = log_dir / run_name
        
        algorithm.train(run_dir=run_dir)
        
        sb3_agent.learn(total_timesteps=cfg.algorithm.n_iterations * cfg.algorithm.num_steps * cfg.env.num_envs, log_interval=1,
                    callback=WandbCallback(
                        gradient_save_freq=0,
                        verbose=2
                    ))
        
        sb3_agent.save(run_dir / f"sb3.pt")

        agent.load_model(run_dir / f"iter_{cfg.algorithm.n_iterations}.pt", device, algorithm.optimizer)
        agent.eval()

        # --- Custom PPO rollout + video ---
        render_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)
        frames = render_env.record_rollout(agent, 200)
        custom_video_path = run_dir / "videos" / "custom_ppo.mp4"
        save_video(frames, custom_video_path)

        # --- SB3 PPO rollout + video ---
        sb3_env   = NavigationEnvSB3(cfg, num_rays, cfg.env.ray_dim, cfg.env.proprio_dim)
        sb3_agent = PPO.load(run_dir / "sb3.pt", env=sb3_env)

        frames_sb3 = sb3_env.record_rollout(sb3_agent, 200)
        sb3_video_path = run_dir / "videos" / "sb3_ppo.mp4"
        save_video(frames_sb3, sb3_video_path)

        wandb.log({
                    "Custom PPO/Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4"),
                    "SB3/Rollout":        wandb.Video(str(sb3_video_path),   fps=10, format="mp4"),
                })

if __name__ == "__main__":
    main()
