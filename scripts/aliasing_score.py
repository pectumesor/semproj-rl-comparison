"""
Perceptual aliasing score of a room (experiment E1a), no training needed.

Samples random positions and headings, renders the ray observation at each, and reports the
fraction of far-apart position pairs whose observations are nearly identical:

    score = P( ||o_i - o_j|| < eps_frac * median ||o - o'||  |  ||p_i - p_j|| > min_pos_dist )

A high score means the observation alone leaves the agent unsure where it is, so memory should pay off.
Run it per room and colour mode, then plot memory gain (best recurrent - MLP) against the score:

    python scripts/aliasing_score.py env.room_path=rooms/four_room.json env.colour_mode=off
    python scripts/aliasing_score.py --multirun env.room_path=rooms/two_room.json,rooms/C_maze.json \
        env.colour_mode=global,off

Options (hydra overrides, all optional): +aliasing.n_samples=2000 +aliasing.eps_frac=0.1
+aliasing.min_pos_dist=100 +aliasing.seed=0. Results are appended to <storage>/aliasing/scores.jsonl.
"""
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch

import utils  # noqa: F401  -- must come before envs: envs -> utils.geometry -> utils -> algorithms -> envs is circular
from envs import NavigationEnv, compute_num_rays


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    aliasing = cfg.get("aliasing", {})
    n_samples    = int(aliasing.get("n_samples", 2000))
    eps_frac     = float(aliasing.get("eps_frac", 0.1))
    min_pos_dist = float(aliasing.get("min_pos_dist", 100.0))
    seed         = int(aliasing.get("seed", 0))

    torch.manual_seed(seed)
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    ray_dim = np.array([cfg.env.ray_encoding, num_rays])
    env = NavigationEnv(cfg=cfg, agent=None, num_rays=num_rays, obs_dim=ray_dim, num_envs=n_samples)
    env.reset()  # draws the goals (goal_mode) and cue colours

    # Random poses inside the spawn regions; the agent's own last action is not part of the place,
    # so only the rays are compared
    env.agent_pos = env.spawn_sampler.sample(n_samples)
    env.facing_direction = torch.rand(n_samples) * 2 * torch.pi
    obs = env.get_observations(*env.ray_cast.scan(env.agent_pos, env.facing_direction))
    o = obs["rays"].flatten(1)                                                  # (N, C*R)

    obs_dist = torch.cdist(o, o)                                                # (N, N)
    pos_dist = torch.cdist(env.agent_pos, env.agent_pos)
    pairs = torch.triu(torch.ones(n_samples, n_samples, dtype=torch.bool), diagonal=1)

    eps = eps_frac * obs_dist[pairs].median()
    far = pairs & (pos_dist > min_pos_dist)
    aliased = far & (obs_dist < eps)
    score = (aliased.sum() / far.sum().clamp_min(1)).item()

    result = {
        "room": Path(cfg.env.room_path).stem,
        "colour_mode": cfg.env.colour_mode,
        "fov": cfg.env.fov,
        "score": score,
        "n_samples": n_samples, "eps_frac": eps_frac, "min_pos_dist": min_pos_dist, "seed": seed,
    }
    print(json.dumps(result))

    out_dir = Path(cfg.storage) / "aliasing"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scores.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
