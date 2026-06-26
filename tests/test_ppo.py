"""
Compare my own PPO implementation with Stable Baselines

"""
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
from envs import NavigationEnv, compute_num_rays, NavigationEnvSB3, MyBackbone
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
                      actor=actor, critic=crtic).to(device)
    
    buffer = RolloutBuffer(obs_dim=obs_dim,
                           act_dim=cfg.env.act_dim,
                           num_steps=cfg.env.num_steps,
                           num_envs=cfg.env.num_envs,
                           gamma=cfg.env.gamma,
                           gae_lambda=cfg.algorithms.gae_lambda,
                           device=device)
 

    env      = NavigationEnv(cfg, agent, num_rays, obs_dim, cfg.env.num_envs, device=device)
    eval_env = NavigationEnv(cfg, agent, num_rays, obs_dim, 1,               device=device)
    
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
                       eval_env=eval_env
                       )
    
    algorithm.train()


    vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, obs_dim), n_envs=cfg.env.num_envs)

    policy_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.model.backbone_hidden_sizes[-1],
            obs_embed_hidden_sizes=list(cfg.model.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.model.backbone_hidden_sizes),
        ),
        net_arch=dict(pi=list(cfg.model.policy_hidden_sizes),
                      vf=list(cfg.model.value_hidden_sizes)),
        activation_fn=nn.ReLU,
        share_features_extractor=True,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.model.lr,
        n_steps=cfg.env.num_steps,
        batch_size=cfg.model.batch_size,
        n_epochs=cfg.env.n_epochs,
        gamma=cfg.env.gamma,
        gae_lambda=cfg.algorithms.gae_lambda,
        clip_range=cfg.algorithms.clip_epsilon,
        ent_coef=cfg.algorithms.entropy_coeff,
        vf_coef=cfg.algorithms.val_coeff,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.env.n_iterations * cfg.env.num_steps * cfg.env.num_envs)



if __name__ == "__main__":
    main()
