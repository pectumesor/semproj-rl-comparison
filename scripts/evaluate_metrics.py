"""

Load trained model weights and evaluate them custom metrics

Every (observation, backbone) combination is evaluated on every seed, all inside
a single wandb run; results are logged once at the end as one table plus a
scatter plot per architecture.
Overrides:
    +eval.log_root=<dir containing per-seed run folders>  (default: <repo>/logs)
    +eval.run_name=<checkpoint folder inside each seed folder, or "latest">
    +eval.seeds=[0,1,2,3,4]
    +eval.observations=[mlp_observation,cnn_observation]
    +eval.backbones=[mlp_backbone,lstm_backbone]

"""
import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

from utils import (create_ppo_agent, test_evaluate_scatter_plot, log_scatter_plots,
create_buffer, create_algorithm, create_environment)
from envs import compute_num_rays

import torch
import numpy as np

device = torch.device("cpu")

DEFAULT_RUN_NAME = "26_09_23_frame_stack_model"


def compose_architecture_cfg(observation: str, backbone: str) -> DictConfig:
    # Re-compose base with this job's CLI overrides, swapping only the architecture groups
    task_overrides = [o for o in HydraConfig.get().overrides.task
                      if not o.lstrip("+~").startswith(("observation=", "backbone="))]
    return compose(config_name="base",
                   overrides=task_overrides + [f"observation={observation}", f"backbone={backbone}"])


def resolve_run_dir(seed_dir: Path, run_name: str) -> Path:
    if run_name != "latest":
        return seed_dir / run_name
    # Most recently written checkpoint, e.g. right after training
    checkpoints = sorted(seed_dir.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1].parent if checkpoints else seed_dir / run_name


def architecture_name(cfg: DictConfig) -> str:
    return (f"encoder:{cfg.observation.name}_backbone:{cfg.backbone.name}_"
            f"frame_stack:{cfg.observation.frame_stack.enabled}_grid_cell:{cfg.observation.grid_cell.enabled}_"
            f"auxiliary_head:{cfg.head.auxiliary_head.enabled}")


def evaluate_architecture(cfg: DictConfig, log_root: Path, run_name: str, seeds: list, scatter_rows: list):

    base_name = f"{cfg.observation.name}_{cfg.backbone.name}_{cfg.algorithm.name}"

    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    ray_dim = np.array([cfg.env.ray_encoding, num_rays])

    for seed in seeds:
        run_dir = resolve_run_dir(log_root / f"{base_name}_seed_{seed}", run_name)
        if not (run_dir / "best.pt").is_file():
            warnings.warn(f"No checkpoint at {run_dir / 'best.pt'}, skipping {base_name} seed {seed}")
            continue

        # The env reads cfg.seed (env.seed, Perlin color field), so it must match the trained seed
        cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = create_ppo_agent(ray_dim= ray_dim, cfg=cfg).to(device)

        env      = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim,
                                      num_envs=cfg.env.num_envs, device=device)
        eval_env = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim,
                                      num_envs=cfg.env.num_eval_envs, device=device)

        buffer = create_buffer(backbone_type=cfg.backbone.name, algorithm_name=cfg.algorithm.name,
                               ray_dim=ray_dim, proprio_dim=cfg.env.proprio_dim,
                               device=device, cfg=cfg)

        algorithm = create_algorithm(cfg=cfg, algorithm_name=cfg.algorithm.name, backbone_type=cfg.backbone.name,
                                     buffer=buffer, device=device, env=env, eval_env=eval_env, agent=agent)

        agent.load_model(run_dir / "best.pt", device, algorithm.optimizer)
        agent.eval()

        # Evaluate metrics on a fixed starting and ending goal
        env.unwrapped.random_pos_flag = False

        test_evaluate_scatter_plot(agent=agent, env=env,
                                   episodes=cfg.env.completion_rate_eps,
                                   nr_runs=cfg.env.mean_of_means_runs,
                                   json_path=cfg.env.room_path,
                                   backbone_type=cfg.backbone.name,
                                   architecture=base_name,
                                   scatter_rows=scatter_rows)


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    choices = HydraConfig.get().runtime.choices

    log_root = Path(OmegaConf.select(cfg, "eval.log_root", default=str(ROOT_DIR / "logs")))
    run_name = OmegaConf.select(cfg, "eval.run_name", default=DEFAULT_RUN_NAME)
    seeds = list(OmegaConf.select(cfg, "eval.seeds", default=[cfg.seed]))
    observations = list(OmegaConf.select(cfg, "eval.observations", default=[choices["observation"]]))
    backbones = list(OmegaConf.select(cfg, "eval.backbones", default=[choices["backbone"]]))

    arch_cfgs = [compose_architecture_cfg(observation, backbone)
                 for observation in observations for backbone in backbones]

    # Base config for the shared settings, full config of each evaluated architecture under "architectures"
    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config.update({
            "architectures": {architecture_name(arch_cfg): OmegaConf.to_container(arch_cfg, resolve=True)
                              for arch_cfg in arch_cfgs}
        })

    wandb.login()

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=run_config,
                        group=cfg.wandb.group, name=f"eval_{cfg.algorithm.name}_seeds_{'-'.join(map(str, seeds))}",
                        tags=["eval", f"algorithm:{cfg.algorithm.name}", f"room:{cfg.env.room_path}"]
                             + [f"architecture:{arch_cfg.observation.name}_{arch_cfg.backbone.name}" for arch_cfg in arch_cfgs],
                        mode=cfg.wandb.mode):

        scatter_rows = []

        for arch_cfg in arch_cfgs:
            evaluate_architecture(arch_cfg, log_root=log_root, run_name=run_name,
                                  seeds=seeds, scatter_rows=scatter_rows)

        log_scatter_plots(scatter_rows)

if __name__ == "__main__":
    main()
