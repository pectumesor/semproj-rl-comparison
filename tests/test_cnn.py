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
from envs import (NavigationEnv, compute_num_rays, NavigationEnvSB3)
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


@hydra.main( config_path="../configs", config_name="test_recurrent", version_base=None)
def main(cfg: DictConfig):

    wandb.login()

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True),
                     sync_tensorboard=True):

        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        agent = create_ppo_agent("CNN","LSTM", ray_dim, cfg).to(device)

        env   = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=ray_dim,
                                           num_envs=cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, 
                                     obs_dim=ray_dim, num_envs=1, device=device).compile()
        buffer = RolloutBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device,cfg=cfg)


        algorithm = RecuurentPPO(num_layers=cfg.algorithm.num_layers, hidden_size=cfg.algorithm.hidden_size,
                                 num_minibatches=cfg.algorithm.minibatch_size,
                                 buffer=buffer, device=device, env=env, eval_env=eval_env,
                                agent=agent, cfg=cfg)


        log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}_cnn"
        run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
        run_dir = log_dir / run_name


        algorithm.train(run_dir=run_dir)
        
        agent.load_model(run_dir / f"iter_{cfg.algorithm.n_iterations}.pt", device, algorithm.optimizer)
        agent.eval()

        # --- Custom PPO rollout + video ---
        render_env = NavigationEnv(cfg, agent, num_rays, ray_dim, 1, device=device)
        frames = render_env.record_recurrent_rollout(agent,
                                                     cfg.algorithm.num_layers,
                                                     1,
                                                     cfg.algorithm.hidden_size, 200)
        custom_video_path = run_dir / "videos" / "custom_lstm_ppo.mp4"
        save_video(frames, custom_video_path)

        wandb.log({
                    "Custom PPO/Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4"),
                })

if __name__ == "__main__":
    main()
