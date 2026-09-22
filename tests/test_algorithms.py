"""
Compare my custom implementations with Stable Baselines 3

"""
import sys
from datetime import datetime
from pathlib import Path
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
# Architecture pieces

from algorithms import RolloutBuffer, MLPPPO, MLPSAC, ReplayBuffer

#Env
from envs import NavigationEnv, compute_num_rays, NavigationEnvSB3
from stable_baselines3.common.env_util import make_vec_env
from utils import create_agent, create_buffer, create_algorithm, save_video, sb3_load

import torch

device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="test_algorithm", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    wandb.login()

    frame_stack = cfg.observation.frame_stack.enabled
    grid_cell = cfg.observation.grid_cell.enabled
    auxiliary_head = cfg.head.auxiliary_head.enabled

    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config.update({
            "architecture_name": (
                f"encoder:{cfg.observation.name}_backbone:{cfg.backbone.name}_"
                f"frame_stack:{frame_stack}_grid_cell:{grid_cell}_"
                f"auxiliary_head:{auxiliary_head}"
            )
        })

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=run_config,
                        group=cfg.wandb.group, name=f"{cfg.observation.name}_{cfg.backbone.name}_{cfg.algorithm.name}_seed_{cfg.seed}",
                        tags=[f"backbone:{cfg.backbone.name}", f"encoder:{cfg.observation.name}",f"algorithm:{cfg.algorithm.name}",
                                f"frame_stack:{frame_stack}", f"grid_cell:{grid_cell}",
                                f"auxiliary_head:{auxiliary_head}", f"room:{cfg.env.room_path}"],
                        sync_tensorboard=True, reinit=True, mode=cfg.wandb.mode):

        # Get observation dimentions
        num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
        ray_dim = np.array([cfg.env.ray_encoding, num_rays])

        # Set up my own custom agents
        custom_agent = create_agent(sb3_flag=False, cfg=cfg, ray_dim=ray_dim)
        buffer = create_buffer(backbone_type=cfg.backbone.name, algorithm_name=cfg.algorithm.name,
                               ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device,
                               cfg=cfg)
        custom_env   = NavigationEnv(cfg=cfg, agent=custom_agent, num_rays=num_rays, obs_dim=ray_dim,
                                                   num_envs=cfg.env.num_envs, device=device)
        custom_eval_env = NavigationEnv(cfg=cfg, agent=custom_agent, num_rays=num_rays,
                                             obs_dim=ray_dim, num_envs=cfg.env.num_eval_envs, device=device)
        
        custom_algorithm = create_algorithm(cfg=cfg, algorithm_name=cfg.algorithm.name, backbone_type=cfg.backbone.name,
                                             buffer=buffer, device=device, env=custom_env, eval_env=custom_eval_env,
                                             agent=custom_agent)

        log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}"
        run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
        run_dir = log_dir / run_name

        # Set up SB3's Agent
        vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, cfg.env.proprio_dim), n_envs=cfg.env.num_envs)
        sb3_agent = create_agent(sb3_flag=True, cfg=cfg, ray_dim=ray_dim, env=vec_env,
                                 tensorboard_log=str(run_dir / "sb3_tensorboard"))

        custom_algorithm.train(run_dir=run_dir)

        sb3_agent.learn(total_timesteps=cfg.algorithm.n_iterations * cfg.algorithm.env_steps_per_iteration * cfg.env.num_envs)
        sb3_agent.save(run_dir / f"sb3.pt")

        # --- Custom rollout + video ---
        custom_render_env = NavigationEnv(cfg, custom_agent, num_rays, ray_dim, 1, device=device)
        frames = custom_render_env.record_rollout(cfg.backbone.name, custom_agent, 200, cfg)
        custom_video_path = ROOT_DIR / "videos" / f"custom_{cfg.algorithm.name}.mp4"
        save_video(frames, custom_video_path)
        wandb.log({
                    f"Custom Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4")
                    })
    
        # --- SB3 rollout + video ---
        sb3_env   = NavigationEnvSB3(cfg, num_rays, ray_dim, cfg.env.proprio_dim)
        sb3_agent = sb3_load(cfg.algorithm.name, run_dir / "sb3.pt", env=sb3_env)
        frames_sb3 = sb3_env.record_rollout(sb3_agent, 200)
        sb3_video_path = ROOT_DIR / "videos" / f"sb3_{cfg.algorithm.name}.mp4"
        save_video(frames_sb3, sb3_video_path)
        wandb.log({
                            f"SB3 Rollout": wandb.Video(str(sb3_video_path), fps=10, format="mp4")
                            })

if __name__ == "__main__":
    main()
