from models import (MLPObservationEmbeddings, CNNObservationEmbeddings,
                     MLPBackbone, SimpleLSTM, GuassianPolicyHead, ValueNet, PPOAgent,
                      SquashedGaussianPolicyHead, DoubleQNet, SACAgent, RecurrentPPOAgent,
                      FrameStackMLP, FrameStackCNN)

from algorithms import (MLPPPO, RecurrentPPO, MLPSAC, RolloutBuffer, ReplayBuffer, RecurrentRolloutBuffer,
                        FrameStackRolloutBuffer, FrameStackRecurrentRolloutBuffer, FrameStackReplayBuffer)

from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
import torch.nn as nn
import numpy as np
import torch


def create_observation_model(frame_stack: bool, observation_type: str, ray_dim: tuple, cfg: DictConfig):
    if frame_stack:
        if observation_type == "mlp":
            return FrameStackMLP(
                input_dim=np.prod(ray_dim) + cfg.env.proprio_dim,
                stack_depth=cfg.observation.frame_stack.size,
                hidden_sizes=cfg.observation.obs_embed_hidden_sizes,
                out_feature_dim=cfg.observation.obs_embed_hidden_sizes[-1]
                )
        else:

            return FrameStackCNN(stack_depth=cfg.observation.frame_stack.size,
                                ray_channels=cfg.env.ray_encoding,
                                cnn_out_channels=cfg.observation.cnn_output_channels,
                                out_feature_dim=cfg.observation.obs_embed_hidden_sizes[-1]
                                        )
    else:
        if observation_type == "mlp":
            return MLPObservationEmbeddings(
                input_dim=np.prod(ray_dim) + cfg.env.proprio_dim,
                hidden_sizes=cfg.observation.obs_embed_hidden_sizes,
                out_feature_dim=cfg.observation.obs_embed_hidden_sizes[-1]
            )
        else:
            return CNNObservationEmbeddings(ray_channels=cfg.env.ray_encoding,
                                        cnn_out_channels=cfg.observation.cnn_output_channels,
                                        proprio_dim=cfg.env.proprio_dim,
                                        proprio_hidden_sizes=cfg.observation.obs_embed_hidden_sizes,
                                        out_feature_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                                        )
    
def create_environment(cfg: DictConfig, agent, num_rays: int, ray_dim: tuple, num_envs: int, device: str = "cpu"):
    from envs import NavigationEnv, NavigationEnvEasy, FrameStackWrapper  # deferred: envs -> utils.geometry -> utils -> models is circular at module scope

    env_cls = NavigationEnvEasy if cfg.env.name == "NavEnvEasy" else NavigationEnv
    env = env_cls(cfg=cfg, agent=agent, num_rays=num_rays, obs_dim=ray_dim, num_envs=num_envs, device=device)

    if cfg.observation.frame_stack.enabled:
        env = FrameStackWrapper(env, stack_size=cfg.observation.frame_stack.size)

    return env

def create_backbone_model(backbone_type: str, cfg: DictConfig):

    if backbone_type == "mlp":

        return MLPBackbone(input_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                           hidden_sizes=cfg.backbone.backbone_hidden_sizes,
                           output_dim=cfg.backbone.backbone_hidden_sizes[-1])
    elif backbone_type == "lstm":

        return SimpleLSTM(input_dim=cfg.observation.obs_embed_hidden_sizes[-1],
                          feature_dim= cfg.backbone.lstm_backbone_feature_dim,
                          num_layers=cfg.backbone.lstm_num_layers)
    
def create_agent(sb3_flag: bool, cfg: DictConfig, ray_dim: tuple, env = None, tensorboard_log: str = None):

    if sb3_flag: # Create Stable Baselines 3 Agent
        if cfg.algorithm.name == "ppo":
            return create_sb3_ppo_agent(backbone_type=cfg.backbone.name,
                                        cfg=cfg, ray_dim=ray_dim, env=env,
                                        tensorboard_log=tensorboard_log)
        else:
            return create_sb3_sac_agent(cfg=cfg, ray_dim=ray_dim, env=env,
                                        tensorboard_log=tensorboard_log)
    else: # Create my custom agent
        if cfg.algorithm.name == "ppo":
            return create_ppo_agent(observation_type=cfg.observation.name,
                                    backbone_type=cfg.backbone.name, ray_dim=ray_dim,
                                    cfg=cfg)
        else:
            return create_sac_agent(observation_type=cfg.observation.name, ray_dim=ray_dim,
                                    cfg=cfg)
    
def create_ppo_agent(ray_dim: tuple,  cfg: DictConfig):


    observation_model = create_observation_model(frame_stack=cfg.observation.frame_stack.enabled,
                                                observation_type=cfg.observation.name,
                                                ray_dim=ray_dim,
                                                 cfg=cfg
                                                )

    backbone_model = create_backbone_model(backbone_type=cfg.backbone.name,
                                           cfg=cfg)

    actor = GuassianPolicyHead(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.head.policy_hidden_sizes)
        
    critic = ValueNet(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                     hidden_sizes=cfg.head.value_hidden_sizes)
    
    if cfg.backbone.name == "mlp":
        return PPOAgent(obs_embed_model= observation_model,
                        backbone_model= backbone_model,
                        actor=actor, critic=critic,
                        action_low=cfg.env.action_low, action_high=cfg.env.action_high)
    else:
        return RecurrentPPOAgent(obs_embed_model= observation_model,
                                 backbone_model= backbone_model,
                                 actor=actor, critic=critic,
                                 action_low=cfg.env.action_low, action_high=cfg.env.action_high)

def create_sac_agent(observation_type: str, ray_dim:int, cfg: DictConfig):
     
    observation_model = create_observation_model(observation_type=observation_type,
                                                 ray_dim=ray_dim,
                                                 cfg=cfg
                                                )

    backbone_model = create_backbone_model(backbone_type="mlp",
                                           cfg=cfg)
    
    actor = SquashedGaussianPolicyHead(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                                action_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.head.policy_hidden_sizes)
    
    critic = DoubleQNet(backbone_dim=cfg.backbone.backbone_hidden_sizes[-1],
                        action_dim= cfg.env.act_dim,
                        hidden_sizes=cfg.head.value_hidden_sizes)
        
    return SACAgent(obs_embed_model=observation_model,
                    backbone_model=backbone_model,
                    actor=actor, critic=critic,
                    action_low=cfg.env.action_low, action_high=cfg.env.action_high)

def create_sb3_ppo_agent(backbone_type: str, cfg: DictConfig, ray_dim: int, env, tensorboard_log: str = None):

    from envs import MyBackbone  # deferred: envs -> utils.geometry -> utils -> models is circular at module scope
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
    if backbone_type == "mlp":
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

def create_sb3_sac_agent(cfg: DictConfig, ray_dim: int, env, tensorboard_log: str = None):

    from envs import MyBackbone  # deferred: envs -> utils.geometry -> utils -> models is circular at module scope
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
            # gSDE: hold exploration noise fixed for sde_sample_freq steps instead of
            # resampling i.i.d. every env.step(). Per-step-independent noise on this env's
            # instantaneous heading-turn action causes undirected jitter instead of committed
            # travel toward the goal (see diagnosis in conversation); gSDE fixes that directly.
            use_sde=True, sde_sample_freq=cfg.algorithm.train_freq,
            policy_kwargs=sac_policy_kwargs, tensorboard_log=tensorboard_log, verbose=1,
        )

def sb3_load(algorithm_type: str, path: str, env):

    if algorithm_type == "ppo":
       return PPO.load(path, env=env)
    else:
        return SAC.load(path, env=env)


def create_algorithm(cfg: DictConfig, algorithm_name: str, backbone_type: str, buffer, device, env, eval_env, agent):

    if algorithm_name == "sac":
        return MLPSAC(buffer, device, env, eval_env, agent, cfg)

    if backbone_type == "mlp":
        return MLPPPO(buffer, device, env, eval_env, agent, cfg)
    else:
        return RecurrentPPO(num_layers=cfg.backbone.lstm_num_layers, hidden_size=cfg.backbone.lstm_backbone_feature_dim,
                                     minibatch_size=cfg.algorithm.minibatch_size,
                                     bptt_window=cfg.backbone.bptt_window_size,
                                     buffer=buffer, device=device, env=env, eval_env=eval_env,
                                    agent=agent, cfg=cfg)

def create_buffer(backbone_type: str, algorithm_name:str,
                  ray_dim: tuple, proprio_dim: int, device: torch.device, cfg: DictConfig):

    frame_stack = cfg.observation.frame_stack.enabled
    stack_size  = cfg.observation.frame_stack.size

    if algorithm_name == "ppo":
        if backbone_type == "mlp":
            if frame_stack:
                return FrameStackRolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, stack_size=stack_size, device=device, cfg=cfg)
            return RolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device,cfg=cfg)
        else:
            if frame_stack:
                return FrameStackRecurrentRolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, stack_size=stack_size, device=device, cfg=cfg)
            return RecurrentRolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device, cfg=cfg)
    else:
        if frame_stack:
            return FrameStackReplayBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, stack_size=stack_size, device=device, cfg=cfg)
        return ReplayBuffer(ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim, device=device, cfg=cfg)

