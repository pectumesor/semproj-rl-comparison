import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from utils import create_ppo_agent, save_video, create_buffer, create_algorithm
from envs import (NavigationEnv, compute_num_rays)


import torch
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.login()

    for i in range(2):

        with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, 
                        config=OmegaConf.to_container(cfg, resolve=True),
                        name=f"{cfg.observation.name}_{cfg.backbone.name}_{cfg.algorithm.name}_run_{i}",
                        group=cfg.wandb.group,
                        sync_tensorboard=True, reinit=True):

            trial_name = wandb.run.name
            log_dir = ROOT_DIR / "logs" / f"{trial_name}"
            run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
            run_dir = log_dir / run_name
            
            num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
            ray_dim = np.array([cfg.env.ray_encoding, num_rays])

            agent = create_ppo_agent(observation_type=cfg.observation.name, backbone_type=cfg.backbone.name,
                                    ray_dim= ray_dim, cfg=cfg).to(device)

            env   = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=ray_dim,
                                            num_envs=cfg.env.num_envs, device=device).compile()
            eval_env = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, 
                                        obs_dim=ray_dim, num_envs=1, device=device).compile()

            buffer = create_buffer(type=cfg.algorithm.name, ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim,
                                device=device, cfg=cfg)

            algorithm = create_algorithm(cfg=cfg, type=cfg.backbone.name, buffer=buffer, device=device,
                                        env=env, eval_env=eval_env, agent=agent)

            
            algorithm.train(trial_name=trial_name, run_dir=run_dir)            
            agent.load_model(run_dir / f"iter_{cfg.algorithm.n_iterations}.pt", device, algorithm.optimizer)
            agent.eval()

            # --- Custom PPO rollout + video ---
            render_env = NavigationEnv(cfg, agent, num_rays, ray_dim, 1, device=device)
            frames = render_env.record_rollout(cfg.backbone.name, agent, 200, cfg)
            custom_video_path = run_dir / "videos" / f"{trial_name}.mp4"
            save_video(frames, custom_video_path)

            wandb.log({
                        f"Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4")
                    })

if __name__ == "__main__":
    main()
