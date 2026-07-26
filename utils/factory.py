from models import (MLPObservationEmbeddings, CNNObservationEmbeddings,
                     MLPBackbone, SimpleLSTM, GuassianPolicyHead, ValueNet, PPOAgent,
                      SquashedGaussianPolicyHead, DoubleQNet, SACAgent, RecurrentPPOAgent)

from envs import MyBackbone
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
import torch.nn as nn
import numpy as np


def create_observation_model(type: str, ray_dim: int, cfg: DictConfig):

    if type == "MLP":
        return MLPObservationEmbeddings(
            input_dim=np.prod(ray_dim) + cfg.env.proprio_dim,
            hidden_sizes=cfg.model.obs_embed_hidden_sizes,
            feature_dim=cfg.model.obs_embed_hidden_sizes[-1]
            )
    elif type == "CNN":

        return CNNObservationEmbeddings(ray_channels=ray_dim,
                                        cnn_out_channels=cfg.model.cnn_output_channels,
                                        proprio_dim=cfg.env.proprio_dim,
                                        proprio_hidden_sizes=cfg.model.obs_embed_hidden_sizes,
                                        feature_dim=cfg.model.obs_embed_hidden_sizes[-1],
                                        )
    
def create_backbone_model(type: str, cfg: DictConfig):

    if type == "MLP":

        return MLPBackbone(input_dim=cfg.model.obs_embed_hidden_sizes[-1],
                           hidden_sizes=cfg.model.backbone_hidden_sizes,
                           output_dim=cfg.model.backbone_hidden_sizes[-1])
    elif type == "LSTM":

        return SimpleLSTM(input_dim=cfg.model.obs_embed_hidden_sizes[-1],
                          feature_dim= cfg.model.lstm_backbone_feature_dim,
                          num_layers=cfg.model.lstm_num_layers)
    
def create_ppo_agent(observation_type: str, backbone_type: str, ray_dim: tuple,  cfg: DictConfig):


    observation_model = create_observation_model(type=observation_type,
                                                 ray_dim=ray_dim,
                                                 cfg=cfg
                                                )
    
    backbone_model = create_backbone_model(type=backbone_type,
                                           cfg=cfg)
    
    actor = GuassianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.model.policy_hidden_sizes)
        
    critic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                     hidden_sizes=cfg.model.value_hidden_sizes)
    
    if backbone_model == "MLP":
        return PPOAgent(obs_embed_model= observation_model,
                        backbone_model= backbone_model,
                        actor=actor, critic=critic,
                        action_low=cfg.env.action_low, action_high=cfg.env.action_high)
    else:
        return RecurrentPPOAgent(obs_embed_model= observation_model,
                                 backbone_model= backbone_model,
                                 actor=actor, critic=critic,
                                 action_low=cfg.env.action_low, action_high=cfg.env.action_high)

def create_sac_agent(observation_type: str, backbone_type: str, ray_dim:int, cfg: DictConfig):
     
    observation_model = create_observation_model(type=observation_type,
                                                 ray_dim=ray_dim, 
                                                 cfg=cfg
                                                )
    
    backbone_model = create_backbone_model(type=backbone_type,
                                           cfg=cfg)
    
    actor = SquashedGaussianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.model.policy_hidden_sizes)
    
    critic = DoubleQNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                        action_dim= cfg.env.act_dim,
                        hidden_sizes=cfg.model.value_hidden_sizes)
        
    return SACAgent(obs_embed_model=observation_model,
                    backbone_model=backbone_model,
                    actor=actor, critic=critic)

def create_sb3_ppo_agent(type: str, cfg: DictConfig, ray_dim: int, env):

    features_extractor_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.model.backbone_hidden_sizes[-1],
            ray_dim=ray_dim,
            proprio_dim=cfg.env.proprio_dim,
            obs_embed_hidden_sizes=list(cfg.model.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.model.backbone_hidden_sizes),
        ),
        activation_fn=nn.ReLU,
    )

    ppo_policy_kwargs = dict(
        **features_extractor_kwargs,
        net_arch=dict(pi=list(cfg.model.policy_hidden_sizes),
                      vf=list(cfg.model.value_hidden_sizes)),
        share_features_extractor=True,
    )
    if type == "MLP":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.algorithm.lr,
            n_steps=cfg.algorithm.num_steps,
            batch_size=cfg.algorithm.mini_batch_size,
            n_epochs=cfg.algorithm.n_epochs,
            gamma=cfg.env.gamma,
            gae_lambda=cfg.algorithm.gae_lambda,
            clip_range=cfg.algorithm.clip_epsilon,
            ent_coef=cfg.algorithm.entropy_coeff,
            vf_coef=cfg.algorithm.val_coeff,
            policy_kwargs=ppo_policy_kwargs,
            verbose=1,
        )
    else:
        return RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=cfg.algorithm.lr,
            n_steps=cfg.algorithm.num_steps,
            batch_size=cfg.algorithm.mini_batch_size,
            n_epochs=cfg.algorithm.n_epochs,
            gamma=cfg.env.gamma,
            gae_lambda=cfg.algorithm.gae_lambda,
            clip_range=cfg.algorithm.clip_epsilon,
            ent_coef=cfg.algorithm.entropy_coeff,
            vf_coef=cfg.algorithm.val_coeff,
            policy_kwargs=dict(
                **ppo_policy_kwargs,
                lstm_hidden_size=cfg.algorithm.hidden_size,
                n_lstm_layers=cfg.algorithm.num_layers,
            ),
            verbose=1,
        )

def create_sb3_sac_agent(cfg: DictConfig, ray_dim: int, env):
    
    features_extractor_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.model.backbone_hidden_sizes[-1],
            ray_dim=ray_dim,
            proprio_dim=cfg.env.proprio_dim,
            obs_embed_hidden_sizes=list(cfg.model.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.model.backbone_hidden_sizes),
        ),
        activation_fn=nn.ReLU,
    )

    sac_policy_kwargs = dict(
        **features_extractor_kwargs,
        net_arch=dict(pi=list(cfg.model.policy_hidden_sizes),
                      qf=list(cfg.model.value_hidden_sizes)),
        share_features_extractor=True,
    )

    return SAC(
            "MlpPolicy", env, learning_rate=cfg.algorithm.actor_lr, buffer_size=cfg.algorithm.num_steps,
            learning_starts=cfg.algorithm.warm_start_steps, batch_size=cfg.algorithm.mini_batch_size,
            tau=cfg.algorithm.tau, gamma=cfg.env.gamma, train_freq=cfg.algorithm.train_freq,
            gradient_steps=cfg.algorithm.n_gradient_update, target_entropy=cfg.algorithm.target_entropy,
            policy_kwargs=sac_policy_kwargs, verbose=1,
        )
        
