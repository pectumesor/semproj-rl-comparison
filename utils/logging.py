# utils/logging.py
import wandb
from omegaconf import OmegaConf

def init_wandb(cfg):
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),  # logs full hydra config
    )

def log(metrics: dict, step: int):
    wandb.log(metrics, step=step)

def finish():
    wandb.finish()
