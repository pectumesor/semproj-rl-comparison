"""
Compare my own PPO implementation with Stable Baselines

"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from utils import (create_ppo_agent, save_video, create_buffer, 
                   create_algorithm, generate_reference_trajectory, evaluate_model_on_metrics)
from envs import (NavigationEnv, compute_num_rays)


import torch
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )

print(f"My device: {device}")

@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.login()

    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config.update({
        "architecture_name": f"{cfg.observation.name}_{cfg.backbone.name}"
    })

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=run_config,
                    group=cfg.wandb.group, name=f"{cfg.observation.name}_{cfg.backbone.name}_{cfg.algorithm.name}_seed_{cfg.seed}",
                    tags=[f"backbone:{cfg.backbone.name}", f"encoder:{cfg.observation.name}", cfg.algorithm.name],
                    sync_tensorboard=True, reinit=True):

            
        trial_name = wandb.run.name
        log_dir = ROOT_DIR / "logs" / f"{trial_name}"
        run_dir = log_dir / "26_08_model"
        
        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        agent = create_ppo_agent(observation_type=cfg.observation.name, backbone_type=cfg.backbone.name,
                                ray_dim= ray_dim, cfg=cfg).to(device)

        env   = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=ray_dim,
                                        num_envs=cfg.env.num_envs, device=device)
        eval_env = NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, 
                                    obs_dim=ray_dim, num_envs=1, device=device)

        buffer = create_buffer(type=cfg.algorithm.name, ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim,
                            device=device, cfg=cfg)

        algorithm = create_algorithm(cfg=cfg, type=cfg.backbone.name, buffer=buffer, device=device,
                                    env=env, eval_env=eval_env, agent=agent)
        
        agent.load_model(run_dir / f"iter_{cfg.algorithm.n_iterations}.pt", device, algorithm.optimizer)
        agent.eval()

        reference_trajectory = generate_reference_trajectory(cfg.env.room_path)

        evaluate_model_on_metrics(agent=agent, env=eval_env, episodes=10,
                                  nr_runs=10, reference_trajectory=reference_trajectory)

if __name__ == "__main__":
    main()
