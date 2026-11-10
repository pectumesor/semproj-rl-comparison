"""
Log the final per-seed results saved by train.py (auc.npy, success_curve.npy, {cr,spl,dtw,ndtw,npl}_values.npy)
onto the W&B training run of that seed, as "Metrics/..." keys.

train.py does this itself at the end of each run. This script backfills runs that finished before it did
(or whose metrics were recomputed), with the same storage, timestamp and group as the sweep:
    python scripts/log_metrics_to_wandb.py storage=<dir> timestamp=<sweep timestamp> wandb.group=<group>

Runs are matched to their metric folder <storage>/eval_metrics/<timestamp>/<room>/<storage name>/seed_<s>
by their config. Runs that already have the metrics, or that are still running, are skipped.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import wandb

from utils import seed_metric_summary


def storage_name_of(run_config: dict):
    """The metric folder name train.py used for this run; rebuilt for runs logged before it was in the config."""
    if "storage_name" in run_config:
        return run_config["storage_name"]
    backbone = run_config["backbone"]
    recurrent = backbone["name"] != "mlp"
    return (run_config["architecture_name"] + (f"_bptt:{backbone['bptt_window_size']}" if recurrent else "")
            + "_" + run_config["variant_name"])

@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    sweep_dir = Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp)
    runs = wandb.Api().runs(f"{cfg.wandb.entity}/{cfg.wandb.project}", filters={"group": cfg.wandb.group})

    logged = 0
    for run in runs:
        # Training runs of this sweep only (the group also holds the mean_of_means eval run)
        if str(run.config.get("timestamp")) != str(cfg.timestamp) or "variant_name" not in run.config:
            continue
        if run.state == "running":
            print(f"{run.name} ({run.id}) is still running, skipping it")
            continue
        if "Metrics/AUC" in run.summary:
            print(f"{run.name} ({run.id}) already has its metrics")
            continue

        room = Path(run.config["env"]["room_path"]).stem
        seed_dir = sweep_dir / room / storage_name_of(run.config) / f"seed_{run.config['seed']}"
        summary = seed_metric_summary(seed_dir)
        if not summary:
            print(f"No metrics in {seed_dir} for {run.name} ({run.id}), skipping it")
            continue

        # Resume the run to log onto it, with the room in its name as train.py now does
        name = run.name if run.name.startswith(room) else f"{room}_{run.name}"
        with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, id=run.id, resume="must",
                        name=name, mode=cfg.wandb.mode):
            wandb.run.summary.update(summary)
        logged += 1
        print(f"Logged {len(summary)} metrics to {name} ({run.id})")

    print(f"Logged metrics to {logged} runs of group {cfg.wandb.group}")


if __name__ == "__main__":
    main()
