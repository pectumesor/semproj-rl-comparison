"""
Aggregate the per-seed metric arrays written by train.py into mean-of-means plots and log them to W&B.

Run once after a sweep has finished (the .sh scripts do this on the node that launched the sweep),
with the same storage, timestamp and group as the sweep:
    python scripts/evaluate_metrics.py storage=<dir> timestamp=<sweep timestamp> wandb.group=<group>

Every <storage>/eval_metrics/<timestamp>/<room>/<architecture>/ folder becomes one plot,
logged as mean_of_means/<room>/<architecture> and saved next to the seed folders.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import matplotlib
matplotlib.use("Agg")  # No display on the cluster
import matplotlib.pyplot as plt
import wandb

from utils import plot_mean_of_means, MEAN_OF_MEANS_METRICS


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    sweep_dir = Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp)
    architecture_dirs = sorted(p for p in sweep_dir.glob("*/*") if p.is_dir())
    if not architecture_dirs:
        raise FileNotFoundError(f"No <room>/<architecture> metric folders under {sweep_dir}")

    summary_columns = ["room", "architecture", "seeds"] + [
        f"{key}_{stat}" for key, _, _ in MEAN_OF_MEANS_METRICS for stat in ("mean_of_means", "ci95_halfwidth")]
    summary_table = wandb.Table(columns=summary_columns)

    with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, group=cfg.wandb.group,
                    name=f"mean_of_means_{cfg.timestamp}", job_type="eval",
                    config={"storage": str(cfg.storage), "timestamp": str(cfg.timestamp)},
                    tags=["eval", "mean_of_means", f"timestamp:{cfg.timestamp}"],
                    mode=cfg.wandb.mode):

        for architecture_dir in architecture_dirs:
            room, architecture = architecture_dir.parent.name, architecture_dir.name

            fig, summary = plot_mean_of_means(architecture_dir, title=f"{room} | {architecture}")
            if fig is None:
                print(f"No complete seed in {architecture_dir}, skipping it")
                continue

            plot_path = architecture_dir / "mean_of_means.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            wandb.log({f"mean_of_means/{room}/{architecture}": wandb.Image(str(plot_path))})
            summary_table.add_data(room, architecture, ",".join(map(str, summary["seeds"])),
                                   *[summary[key][stat] for key, _, _ in MEAN_OF_MEANS_METRICS
                                     for stat in ("mean_of_means", "ci95_halfwidth")])
            print(f"Logged {plot_path}")

        wandb.log({"mean_of_means/summary": summary_table})


if __name__ == "__main__":
    main()
