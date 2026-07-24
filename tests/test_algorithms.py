"""
Compare my own PPO implementation with Stable Baselines

"""
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
# Architecture pieces

from algorithms import RolloutBuffer, MLPPPO, MLPSAC, ReplayBuffer

#Env
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3
from stable_baselines3.common.env_util import make_vec_env
from utils import create_ppo_agent, create_sac_agent, create_sb3_ppo_agent, create_sb3_sac_agent

import torch

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="test_algorithm", version_base=None)
def main(cfg: DictConfig):

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)

    ray_dim = num_rays * cfg.env.ray_encoding

    if cfg.algorithm.name == "ppo":
        agent = create_ppo_agent("MLP", "MLP", cfg).to(device)

        env      = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1,               device=device).compile()
        vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, cfg.env.proprio_dim), n_envs=cfg.env.num_envs)

        buffer = RolloutBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device,cfg=cfg)
        algorithm = MLPPPO(buffer=buffer,device=device,env=env,eval_env=eval_env, agent=agent, cfg=cfg)
        sb3_agent = create_sb3_ppo_agent(cfg, vec_env)

    else:
        agent = create_sac_agent("MLP", "MLP", cfg).to(device)

        env      = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1,               device=device).compile()
        vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, cfg.env.proprio_dim), n_envs=cfg.env.num_envs)

        buffer = ReplayBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device, cfg=cfg)
        algorithm = MLPSAC(buffer=buffer, device=device,env= env, eval_env=eval_env, agent=agent,cfg=cfg)
        sb3_agent = create_sb3_sac_agent(cfg, vec_env)
        

    log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}"
    run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
    run_dir = log_dir / run_name
    
    algorithm.train(run_dir=run_dir)

    sb3_agent.learn(total_timesteps=cfg.algorithm.n_iterations * cfg.algorithm.num_steps * cfg.env.num_envs)

    sb3_agent.save(run_dir / f"sb3.pt")

if __name__ == "__main__":
    main()
