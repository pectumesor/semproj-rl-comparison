"""
Load trained models and render rollouts. Optionally compile frames into a video.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn

from models import MLPObservationEmbeddings, MLPBackbone, GuassianPolicyHead, ValueNet
from models import PPOAgent
from algorithms import RolloutBuffer, MLPPPO
from envs import NavigationEnvEasy, compute_num_rays, NavigationEnvSB3, MyBackbone
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

device = torch.device("cpu")
print(f"Using device: {device}")

def save_video(frames: list, path: Path, fps: int = 10):
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"Saved video to {path}")


@hydra.main(config_path="../configs", config_name="test_algorithm", version_base=None)
def main(cfg: DictConfig):

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)

    _probe = NavigationEnvEasy(cfg, None, num_rays, (1, num_rays), num_envs=1)
    _obs, _ = _probe.reset()
    ray_dim     = tuple(_obs["rays"].shape[1:])
    proprio_dim = _obs["proprio"].shape[-1]
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
    actor  = GuassianPolicyHead(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                                actions_dim=cfg.env.act_dim,
                                hidden_sizes=cfg.model.policy_hidden_sizes)
    critic = ValueNet(backbone_dim=cfg.model.backbone_hidden_sizes[-1],
                      hidden_sizes=cfg.model.value_hidden_sizes)

    agent = PPOAgent(obs_embed_model=observation_model, backbone_model=backbone_model,
                     actor=actor, critic=critic,
                     action_low=cfg.env.action_low, action_high=cfg.env.action_high).to(device)

    buffer    = RolloutBuffer(ray_dim=ray_dim, proprio_dim=proprio_dim, device=device, cfg=cfg)
    env       = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)
    eval_env  = NavigationEnvEasy(cfg, agent, num_rays, ray_dim, 1, device=device)
    algorithm = MLPPPO(buffer=buffer, device=device, env=env, eval_env=eval_env,
                       agent=agent, cfg=cfg)

    log_dir  = ROOT_DIR / "logs" / "ppo"
    run_name = "26_06_26_15_44_30_model"
    run_dir  = log_dir / run_name

    agent.load_model(run_dir / "iter_100.pt", device, algorithm.optimizer)
    agent.eval()

    # --- Custom PPO rollout + video ---
    frames_custom = []
    obs, _ = env.reset()
    for _ in range(200):
        action = agent.predict_action(obs)
        obs, _, done, _, _ = env.step(action)
        frame = env.record_frame(obs)
        print(f"Frame shape: {frame.shape} ")
        print(f"Frame: {frame}")
        
        frames_custom.append(frame)
        #env.render(obs, "Custom PPO")
        if done.any():
            obs, _ = env.reset()

    save_video(frames_custom, ROOT_DIR / "videos" / "custom_ppo.mp4")

    # --- SB3 PPO rollout + video ---
    vec_env = make_vec_env(lambda: NavigationEnvSB3(cfg, num_rays, ray_dim, proprio_dim), n_envs=1)
    model   = PPO.load(run_dir / "sb3.pt", env=vec_env)

    frames_sb3 = []
    obs = vec_env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, _, done, _ = vec_env.step(action)
        frames_sb3.append(vec_env.envs[0].unwrapped.record_frame(obs[0]))
        #vec_env.envs[0].unwrapped.render({"rays": torch.tensor(obs)}, "SB3 PPO")
        if done:
            obs = vec_env.reset()

    save_video(frames_sb3, ROOT_DIR / "videos" / "sb3_ppo.mp4")


if __name__ == "__main__":
    main()
