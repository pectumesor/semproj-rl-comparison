"""
Load trained models and render rollouts. Optionally compile frames into a video.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import torch

from algorithms import RolloutBuffer, MLPPPO
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3
from stable_baselines3 import PPO
from utils import save_video, create_ppo_agent

device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main(config_path="../configs", config_name="test_algorithm", version_base=None)
def main(cfg: DictConfig):

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    
    agent = create_ppo_agent("MLP", "MLP", cfg).to(device)

    buffer    = RolloutBuffer(ray_dim=cfg.env.ray_dim, proprio_dim=cfg.env.proprio_dim, device=device, cfg=cfg)
    env       = NavigationEnvEasy(cfg, agent, num_rays, cfg.env.ray_dim, 1, device=device)
    eval_env  = NavigationEnvEasy(cfg, agent, num_rays, cfg.env.ray_dim, 1, device=device)
    algorithm = MLPPPO(buffer=buffer, device=device, env=env, eval_env=eval_env,
                       agent=agent, cfg=cfg)

    log_dir  = ROOT_DIR / "logs" / "ppo"
    run_name = "26_06_26_15_44_30_model"
    run_dir  = log_dir / run_name

    agent.load_model(run_dir / "iter_100.pt", device, algorithm.optimizer)
    agent.eval()

    # --- Custom PPO rollout + video ---
    frames = env.record_rollout(agent, 200)
    save_video(frames, ROOT_DIR / "videos" / "custom_ppo.mp4")

    # --- SB3 PPO rollout + video ---
    sb3_env   = NavigationEnvSB3(cfg, num_rays, cfg.env.ray_dim, cfg.env.proprio_dim)
    sb3_agent = PPO.load(run_dir / "sb3.pt", env=sb3_env)

    frames_sb3 = sb3_env.record_rollout(sb3_agent, 200)
    save_video(frames_sb3, ROOT_DIR / "videos" / "sb3_ppo.mp4")


if __name__ == "__main__":
    main()
