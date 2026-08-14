from models import (MLPObservationEmbeddings, CNNObservationEmbeddings,
                     MLPBackbone, SimpleLSTM, GuassianPolicyHead, ValueNet, PPOAgent,
                      SquashedGaussianPolicyHead, DoubleQNet, SACAgent, RecurrentPPOAgent)

from algorithms import MLPPPO, RecurrentPPO, RolloutBuffer, ReplayBuffer

from envs import MyBackbone
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
import torch.nn as nn
import numpy as np
import torch


def create_observation_model(type: str, ray_dim: tuple, cfg: DictConfig):

    if type == "mlp":
        return MLPObservationEmbeddings(
            input_dim=np.prod(ray_dim) + cfg.env.proprio_dim,
            hidden_sizes=cfg.observation.obs_embed_hidden_sizes,
            feature_dim=cfg.observation.obs_embed_hidden_sizes[-1]
            )
    elif type == "cnn":

        return CNNObservationEmbeddings(ray_channels=cfg.env.ray_encoding,
                                        cnn_out_channels=cfg.observation.cnn_output_channels,
                                        proprio_dim=cfg.env.proprio_dim,
                                        proprio_hidden_sizes=cfg.observation.obs_embed_hidden_sizes,
                                        feature_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                                        )
    
def create_backbone_model(type: str, cfg: DictConfig):

    if type == "mlp":

        return MLPBackbone(input_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                           hidden_sizes=cfg.backbone.backbone_hidden_sizes,
                           output_dim=cfg.backbone.backbone_hidden_sizes[-1])
    elif type == "lstm":

        return SimpleLSTM(input_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                          feature_dim= cfg.backbone.lstm_backbone_feature_dim,
                          num_layers=cfg.backbone.lstm_num_layers)
    
def create_ppo_agent(observation_type: str, backbone_type: str, ray_dim: tuple,  cfg: DictConfig):


    observation_model = create_observation_model(type=observation_type,
                                                 ray_dim=ray_dim,
                                                 cfg=cfg
                                                )
    
    backbone_model = create_backbone_model(type=backbone_type,
                                           cfg=cfg)
    
    actor = GuassianPolicyHead(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.head.policy_hidden_sizes)
        
    critic = ValueNet(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                     hidden_sizes=cfg.head.value_hidden_sizes)
    
    if backbone_type == "mlp":
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
    
    actor = SquashedGaussianPolicyHead(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.head.policy_hidden_sizes)
    
    critic = DoubleQNet(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                        action_dim= cfg.env.act_dim,
                        hidden_sizes=cfg.head.value_hidden_sizes)
        
    return SACAgent(obs_embed_model=observation_model,
                    backbone_model=backbone_model,
                    actor=actor, critic=critic)

def create_sb3_ppo_agent(type: str, cfg: DictConfig, ray_dim: int, env, tensorboard_log: str = None):

    features_extractor_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.backbone.backbone_hidden_sizes[-1],
            ray_dim=ray_dim,
            proprio_dim=cfg.env.proprio_dim,
            obs_embed_hidden_sizes=list(cfg.observation.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.backbone.backbone_hidden_sizes),
        ),
        activation_fn=nn.ReLU,
    )

    ppo_policy_kwargs = dict(
        **features_extractor_kwargs,
        net_arch=dict(pi=list(cfg.head.policy_hidden_sizes),
                      vf=list(cfg.head.value_hidden_sizes)),
        share_features_extractor=True,
    )
    if type == "mlp":
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
            tensorboard_log=tensorboard_log,
            seed=cfg.seed,
            verbose=1,
        )
    else:

        return SB3RecurrentPPO(
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
                lstm_hidden_size=cfg.backbone.lstm_backbone_feature_dim,
                n_lstm_layers=cfg.backbone.lstm_num_layers,
            ),
            tensorboard_log=tensorboard_log,
            seed=cfg.seed,
            verbose=1,
        )

def create_sb3_sac_agent(cfg: DictConfig, ray_dim: int, env):
    
    features_extractor_kwargs = dict(
        features_extractor_class=MyBackbone,
        features_extractor_kwargs=dict(
            features_dim=cfg.backbone.backbone_hidden_sizes[-1],
            ray_dim=ray_dim,
            proprio_dim=cfg.env.proprio_dim,
            obs_embed_hidden_sizes=list(cfg.observation.obs_embed_hidden_sizes),
            backbone_hidden_sizes=list(cfg.backbone.backbone_hidden_sizes),
        ),
        activation_fn=nn.ReLU,
    )

    sac_policy_kwargs = dict(
        **features_extractor_kwargs,
        net_arch=dict(pi=list(cfg.head.policy_hidden_sizes),
                      qf=list(cfg.head.value_hidden_sizes)),
        share_features_extractor=True,
    )

    return SAC(
            "MlpPolicy", env, learning_rate=cfg.algorithm.actor_lr, buffer_size=cfg.algorithm.num_steps,
            learning_starts=cfg.algorithm.warm_start_steps, batch_size=cfg.algorithm.mini_batch_size,
            tau=cfg.algorithm.tau, gamma=cfg.env.gamma, train_freq=cfg.algorithm.train_freq,
            gradient_steps=cfg.algorithm.n_gradient_update, target_entropy=cfg.algorithm.target_entropy,
            policy_kwargs=sac_policy_kwargs, verbose=1,
        )
        
def create_algorithm(cfg: DictConfig, type: str, buffer, device, env, eval_env, agent):

    if type == "ppo":
        return MLPPPO(buffer, device, env, eval_env, agent, cfg)
    else:
        return RecurrentPPO(num_layers=cfg.algorithm.num_layers, hidden_size=cfg.algorithm.hidden_size,
                                     num_minibatches=cfg.algorithm.minibatch_size,
                                     buffer=buffer, device=device, env=env, eval_env=eval_env,
                                    agent=agent, cfg=cfg)

def create_buffer(type: str, ray_dim: tuple, proprio_dim: int, device: torch.device, cfg: DictConfig):

    if type == "ppo":
        return RolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device,cfg=cfg)
    else:
        return ReplayBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device, cfg=cfg)

