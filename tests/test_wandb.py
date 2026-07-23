"""
Compare my own PPO implementation with Stable Baselines

"""
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
# Architecture pieces
from models import (MLPObservationEmbeddings,
                    MLPBackbone, GuassianPolicyHead,
                    ValueNet)
# Agents
from models import PPOAgent
from algorithms import RolloutBuffer, MLPPPO

#Env
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3, MyBackbone
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device( "mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available()
                      else "cpu" )
#device = torch.device("cpu")
print(f"Using device: {device}")


@hydra.main( config_path="../configs", config_name="test_wandb", version_base=None)
def main(cfg: DictConfig):

    wandb.login()

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True),
                     sync_tensorboard=True):

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
            
        critic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                        hidden_sizes=cfg.model.value_hidden_sizes)

        agent = PPOAgent(obs_embed_model=observation_model,
                        backbone_model=backbone_model,
                        actor=actor, critic=critic, action_low=cfg.env.action_low, action_high=cfg.env.action_high).to(device)

        env      = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, cfg.env.num_envs, device=device).compile()
        eval_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1,               device=device).compile()
        buffer = RolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device,cfg=cfg)
        algorithm = MLPPPO(buffer=buffer,device=device,env=env,eval_env=eval_env, agent=agent, cfg=cfg)

        vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, proprio_dim), n_envs=cfg.env.num_envs)

        _features_extractor_kwargs = dict(
            features_extractor_class=MyBackbone,
            features_extractor_kwargs=dict(
                features_dim=cfg.model.backbone_hidden_sizes[-1],
                ray_dim=ray_dim,
                proprio_dim=proprio_dim,
                obs_embed_hidden_sizes=list(cfg.model.obs_embed_hidden_sizes),
                backbone_hidden_sizes=list(cfg.model.backbone_hidden_sizes),
            ),
            activation_fn=nn.ReLU,
        )
        ppo_policy_kwargs = dict(
            **_features_extractor_kwargs,
            net_arch=dict(pi=list(cfg.model.policy_hidden_sizes),
                        vf=list(cfg.model.value_hidden_sizes)),
            share_features_extractor=True,
        )

        model = PPO(
        "MlpPolicy",
        vec_env,
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
        tensorboard_log=wandb.run.dir,
        verbose=1,
        )

        log_dir = ROOT_DIR / "logs" / f"{cfg.algorithm.name}"
        run_name = datetime.now().strftime("%y_%m_%d_%H_%M_%S_model")
        run_dir = log_dir / run_name
        
        algorithm.train(run_dir=run_dir)

        model.learn(total_timesteps=cfg.algorithm.n_iterations * cfg.algorithm.num_steps * cfg.env.num_envs, log_interval=1,
                    callback=WandbCallback(
                        gradient_save_freq=0,
                        verbose=2
                    ))

        model.save(run_dir / f"sb3.pt")

        agent.load_model(run_dir / "iter_100.pt", device, algorithm.optimizer)
        agent.eval()

        import imageio
        video_dir =  run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        render_env = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)  # Compile breaks render function
        # --- Custom PPO rollout + video ---
        frames_custom = []
        obs, _ = render_env.reset()
        for _ in range(200):
            action = agent.predict_action(obs)
            obs, _, done, _, _ = render_env.step(action)
            frames_custom.append(render_env.record_frame(obs))
            if done.any():
                obs, _ = render_env.reset()

        custom_video_path = video_dir / "custom_ppo.mp4"
        imageio.mimsave(str(custom_video_path), frames_custom, fps=10)

        # --- SB3 PPO rollout + video ---
        render_vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, proprio_dim), n_envs=1)

        model   = PPO.load(run_dir / "sb3.pt", env=vec_env)

        frames_sb3 = []
        obs = render_vec_env.reset()
        for _ in range(200):
            action, _ = model.predict(obs)
            obs, _, done, _ = render_vec_env.step(action)
            frames_sb3.append(render_vec_env.envs[0].unwrapped.record_frame(obs[0]))
            if done:
                obs = render_vec_env.reset()

        sb3_video_path = video_dir / "sb3_ppo.mp4"
        imageio.mimsave(str(sb3_video_path), frames_sb3, fps=10)

        wandb.log({
                    "Custom PPO/Rollout": wandb.Video(str(custom_video_path), fps=10, format="mp4"),
                    "SB3/Rollout":        wandb.Video(str(sb3_video_path),   fps=10, format="mp4"),
                })


if __name__ == "__main__":
    main()
