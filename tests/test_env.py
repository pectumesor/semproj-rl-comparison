import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from envs import NavigationEnv, compute_num_rays
from gymnasium.vector import AsyncVectorEnv


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

   
def make_env(cfg, agent, num_rays, obs_dim):
        def _init():
            return NavigationEnv(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=obs_dim)
        return _init

@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    num_classes = cfg.env.num_classes
    obs_dim  = (num_classes + 1, num_rays)


    observation_model = MLPObservationEmbeddings(
        input_dim=(num_classes + 1) * num_rays,
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
                      actor=actor, critic=crtic)
    
    buffer = RolloutBuffer(obs_dim=obs_dim,
                           act_dim=cfg.env.act_dim,
                           num_steps=cfg.env.num_steps,
                           num_envs=cfg.env.num_envs,
                           gamma=cfg.env.gamma,
                           gae_lambda=cfg.algorithms.gae_lambda,
                           device=cfg.env.device)
 

    env = AsyncVectorEnv([make_env(cfg, agent, num_rays, obs_dim) for _ in range(cfg.env.num_envs)])
    eval_env = AsyncVectorEnv([make_env(cfg, agent, num_rays, obs_dim)])
    
    algorithm = MLPPPO(buffer=buffer,
                       device=cfg.env.device,
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
                       eval_env=eval_env
                       )
    
    algorithm.train()


if __name__ == "__main__":
    main()
