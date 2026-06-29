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
from models import (MLPObservationEmbeddings,
                    MLPBackbone, GuassianPolicyHead,
                    ValueNet)
# Agents
from models import BaseAgent
from algorithms import RolloutBuffer, MLPPPO

#Env
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3, MyBackbone
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

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

    crtic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                     hidden_sizes=cfg.model.value_hidden_sizes)

    agent = BaseAgent(obs_embed_model=observation_model,
                      backbone_model=backbone_model,
                      actor=actor, critic=crtic).to(device)

    buffer = RolloutBuffer(ray_dim=ray_dim,
                           proprio_dim=proprio_dim,
                           act_dim=cfg.env.act_dim,
                           num_steps=cfg.env.num_steps,
                           num_envs=cfg.env.num_envs,
                           gamma=cfg.env.gamma,
                           gae_lambda=cfg.algorithms.gae_lambda,
                           device=device)

    env      = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)
    eval_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)
    
    algorithm = MLPPPO(buffer=buffer,
                       device=device,
                       env=env,
                       lr=cfg.model.lr,
                       n_iterations=cfg.env.n_iterations,
                       mini_batch=cfg.model.batch_size,
                       n_epochs=cfg.env.n_epochs,
                       agent=agent,
                       gamma=cfg.env.gamma,
                       gae_lambda=cfg.algorithms.gae_lambda,
                       clip_epsilon=cfg.algorithms.clip_epsilon,
                       entropy_coeff=cfg.algorithms.entropy_coeff,
                       val_coeff=cfg.algorithms.val_coeff,
                       aux_coeff=cfg.algorithms.aux_coeff,
                       task_coeff=cfg.algorithms.task_coeff,
                       intr_coeff=cfg.algorithms.intr_coeff,
                       eval_env=eval_env,
                       save_interval=cfg.model.save_interval
                       )
    
    log_dir = ROOT_DIR / "logs" / "ppo"
    run_name = "26_06_26_15_44_30_model"
    run_dir = log_dir / run_name
    
    agent.load_model(run_dir/"iter_100.pt", device, algorithm.optimizer)

    vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, proprio_dim), n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.model.backbone_hidden_sizes[-1],
            ray_dim=ray_dim,
            proprio_dim=proprio_dim,
            obs_embed_hidden_sizes=list(cfg.model.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.model.backbone_hidden_sizes),
        ),
        net_arch=dict(pi=list(cfg.model.policy_hidden_sizes),
                      vf=list(cfg.model.value_hidden_sizes)),
        activation_fn=nn.ReLU,
        share_features_extractor=True,
    )

    model = PPO.load(run_dir/"sb3.pt", env=vec_env)

    obs=vec_env.reset()

    total_reward = 0
    for _ in range(100):
        action,_ = model.predict(obs)
        obs, reward, done,_ = vec_env.step(action)

        if done:
            obs = vec_env.reset()
        vec_env.envs[0].unwrapped.render(obs[0], "SB3 Result")

    total_reward = 0
    agent.eval()
    obs, info = env.reset()
    for _ in range(100):
        action = agent.predict_action(obs)
        obs, reward, done,_, info = env.step(action)
        total_reward += reward

        if done.any():
            obs, _ = env.reset()
        env.render(obs, "Custom Algorithm Result")    


if __name__ == "__main__":
    main()
