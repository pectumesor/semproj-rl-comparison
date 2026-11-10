"""
Aggregate the per-seed metric arrays written by train.py into mean-of-means and IQM plots and log them to W&B.

Run once after a sweep has finished (the .sh scripts do this on the node that launched the sweep),
with the same storage, timestamp and group as the sweep:
    python scripts/evaluate_metrics.py storage=<dir> timestamp=<sweep timestamp> wandb.group=<group>

Every <storage>/eval_metrics/<timestamp>/<room>/<architecture>/ folder becomes one plot,
logged as mean_of_means/<room>/<architecture> (or a short digest of it, see image_key) and saved next to the seed folders.
The IQM plots compare the architectures: one per room and one over all rooms, logged as iqm/<room> and iqm/all_rooms
and saved as <storage>/eval_metrics/<timestamp>/iqm_<room>.png, with their numbers in the iqm/summary table.
The learning curves (success_curve.npy) are logged per room as curves/<room>, one panel per architecture, and as
curves/<room>_interactive, and saved as curves_<room>.png. They are padded to the budget like the AUC, so runs
that stopped early stay comparable.
"""
import sys
import hashlib
import textwrap
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import matplotlib
matplotlib.use("Agg")  # No display on the cluster
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import wandb

from utils import plot_mean_of_means, MEAN_OF_MEANS_METRICS
from utils.metrics import load_seed_metrics
from algorithms.early_stopping import EarlyStopping
from e4_promotion import iqm, stratified_bootstrap_ci, parse_storage_name

# np.trapezoid only exists from numpy 2.0 on, environment.yaml pins numpy<2.0 (np.trapz)
trapezoid = getattr(np, "trapezoid", None) or np.trapz

IQM_METRICS = [("auc", "Success AUC", (0.0, 1.0)), ("best_ema", "Best eval success EMA", (0.0, 1.0))] + MEAN_OF_MEANS_METRICS
IQM_N_BOOT = 2000
CURVE_PANELS_PER_ROW = 5


def image_key(architecture: str):
    """
    W&B copies each image to media/images/<key>_<step>_<hash>.png (~30 characters on top of the key), and
    the storage names of the E4 aux configs (~215 characters) exceed the 255-character file name limit.
    Long names are logged under a digest instead, with the full name in the image caption and the summary table
    """
    return architecture if len(architecture) <= 200 else hashlib.sha1(architecture.encode()).hexdigest()[:12]


def auc_mean_of_means(architecture_dir: Path):
    """Mean of the per-seed budget-padded AUCs (auc.npy) with its 95% CI half-width across seeds (t-distribution)."""
    aucs = np.array([np.load(p)[0] for p in sorted(architecture_dir.glob("seed_*/auc.npy"))])
    if len(aucs) == 0:
        return float("nan"), float("nan")
    half = t.ppf(0.975, len(aucs) - 1) * aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else float("nan")
    return float(aucs.mean()), float(half)

def seed_curves(architecture_dir: Path):
    """The non-empty eval success curves of an architecture's seeds, {seed: curve}."""
    paths = sorted(architecture_dir.glob("seed_*/success_curve.npy"), key=lambda p: int(p.parent.name.split("_")[-1]))
    curves = {int(p.parent.name.split("_")[-1]): np.load(p) for p in paths}
    return {seed: curve for seed, curve in curves.items() if len(curve) > 0}

def best_ema(curve, early_stopping):
    """The best EMA of an eval success curve, recomputed as during training: runs that never stopped early get one too."""
    stopper = EarlyStopping(patience=1, min_delta=early_stopping.min_delta, ema_alpha=early_stopping.ema_alpha)
    for value in curve:
        stopper.step(float(value))
    return stopper.best

def seed_scores(architecture_dir: Path, key: str, early_stopping):
    """
    One score per seed: its auc.npy, the best EMA of its learning curve, or the mean of its runs. Seeds that
    never reached the goal have no DTW/NPL and are left out, so those two panels only cover the seeds that got there
    """
    if key == "auc":
        return np.array([np.load(p)[0] for p in sorted(architecture_dir.glob("seed_*/auc.npy"))])
    if key == "best_ema":
        return np.array([best_ema(curve, early_stopping) for curve in seed_curves(architecture_dir).values()])
    per_seed = load_seed_metrics(architecture_dir)
    return np.array([metrics[key].mean() for metrics in per_seed.values() if len(metrics[key]) > 0])

def iqm_labels(architectures):
    """Row labels: the storage name fields that differ from their most common value in the sweep."""
    fields = {architecture: parse_storage_name(architecture) for architecture in architectures}
    common = {key: Counter(f.get(key) for f in fields.values()).most_common(1)[0][0]
              for f in fields.values() for key in f}
    return {architecture: " ".join(f"{key}:{value}" for key, value in f.items() if value != common[key])
            or "most common settings" for architecture, f in fields.items()}

def iqm_stats(architecture_dirs, rng, early_stopping):
    """
    {architecture: {metric: (IQM, CI low, CI high)}} over all the runs of each architecture, with the
    stratified bootstrap of the promotion rule (seeds resampled within each room).
    architecture_dirs: {architecture: [its folder in each room]}
    """
    stats = {}
    for architecture, dirs in architecture_dirs.items():
        stats[architecture] = {}
        for key, _, _ in IQM_METRICS:
            per_room = [scores for scores in (seed_scores(d, key, early_stopping) for d in dirs) if len(scores) > 0]
            if per_room:
                stats[architecture][key] = (iqm(np.concatenate(per_room)),
                                            *stratified_bootstrap_ci(per_room, IQM_N_BOOT, rng))
    return stats

def plot_iqm(stats, labels, title):
    """One row per architecture (best AUC on top), one panel per metric, IQM with its 95% CI."""
    order = sorted(stats, key=lambda architecture: stats[architecture].get("auc", (-1.0,))[0], reverse=True)
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, len(IQM_METRICS), figsize=(3 * len(IQM_METRICS), 1.3 + 0.3 * len(order)), sharey=True)
    for ax, (key, name, bounds) in zip(axes, IQM_METRICS):
        rows = [(row, stats[architecture][key]) for row, architecture in zip(y, order) if key in stats[architecture]]
        for row, (center, low, high) in rows:
            ax.plot([low, high], [row, row], color="crimson", lw=2, solid_capstyle="round")
            ax.plot(center, row, "o", color="crimson", ms=5)
        if bounds is not None:
            pad = 0.05 * (bounds[1] - bounds[0])
            ax.set_xlim(bounds[0] - pad, bounds[1] + pad)
        else:
            # DTW/NPL have no upper bound: a config that rarely reaches the goal can sit orders of magnitude out
            ax.set_xscale("log")
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="both", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y, [labels[architecture] for architecture in order], fontsize=8)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    fig.suptitle(f"{title} | IQM with 95% stratified-bootstrap CI", fontsize=11)
    fig.tight_layout()
    return fig

def budget_evaluations(n_iterations, eval_interval):
    """Evaluations of a run that never stops early, see MLPPPO.budget_eval_steps."""
    return n_iterations // eval_interval + int(n_iterations % eval_interval != 0)

def padded_curves(curves, budget):
    """(seeds, budget) array of curves, each held at its last value after it stopped, the same padding as the AUC."""
    return np.stack([np.pad(curve, (0, budget - len(curve)), mode="edge") for curve in curves.values()])

def plot_curves(room_curves, labels, title, iterations):
    """
    One panel per architecture (best AUC first) on shared axes: seeds thin with a dot where each stopped early,
    IQM over seeds thick, the IQM of every other architecture in grey behind it.
    room_curves: {architecture: {seed: curve}} (architecture sorted), iterations: x value of each evaluation
    """
    budget = len(iterations)
    seeds = {architecture: padded_curves(curves, budget) for architecture, curves in room_curves.items()}
    centers = {architecture: np.apply_along_axis(iqm, 0, runs) for architecture, runs in seeds.items()}

    columns = min(CURVE_PANELS_PER_ROW, len(seeds))
    rows = int(np.ceil(len(seeds) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.9 * columns, 1.2 + 2.2 * rows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(seeds):]:
        ax.set_visible(False)

    for ax, (architecture, runs) in zip(axes.flat, seeds.items()):
        for other, center in centers.items():
            if other != architecture:
                ax.plot(iterations, center, color="0.85", lw=0.7, zorder=1)
        for curve, run in zip(room_curves[architecture].values(), runs):
            ax.plot(iterations, run, color="crimson", lw=0.6, alpha=0.45, zorder=2)
            if len(curve) < budget:
                ax.plot(iterations[len(curve) - 1], curve[-1], "o", color="crimson", ms=2.5, alpha=0.7, zorder=3)
        ax.plot(iterations, centers[architecture], color="crimson", lw=2, zorder=4)
        ax.set_title(textwrap.fill(labels[architecture], 34) + f"\nAUC {iqm(trapezoid(runs, axis=1) / (budget - 1)):.2f}", fontsize=7.5)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supxlabel("Iteration", fontsize=9)
    fig.supylabel("Eval success rate", fontsize=9)
    fig.suptitle(f"{title} | Learning curves, each seed held at its last value after early stopping", fontsize=11)
    fig.tight_layout()
    return fig

def log_curve_plots(sweep_dir: Path, architecture_dirs, n_iterations, eval_interval):
    """Log the learning curves of every room, as a small-multiples image and as an interactive W&B line plot."""
    rooms = sorted({d.parent.name for d in architecture_dirs})
    labels = iqm_labels({d.name for d in architecture_dirs})
    for room in rooms:
        curves = {d.name: seed_curves(d) for d in architecture_dirs if d.parent.name == room}
        curves = {architecture: seed_curve for architecture, seed_curve in curves.items() if seed_curve}
        if not curves:
            continue
        budget = max([budget_evaluations(n_iterations, eval_interval)] + [len(c) for cs in curves.values() for c in cs.values()])
        iterations = np.minimum(eval_interval * np.arange(1, budget + 1), n_iterations)
        order = sorted(curves, key=lambda a: iqm(trapezoid(padded_curves(curves[a], budget), axis=1)), reverse=True)

        fig = plot_curves({a: curves[a] for a in order}, labels, room, iterations)
        plot_path = sweep_dir / f"curves_{room}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        wandb.log({
            f"curves/{room}": wandb.Image(str(plot_path), caption=room),
            f"curves/{room}_interactive": wandb.plot.line_series(
                xs=iterations.tolist(), ys=[np.apply_along_axis(iqm, 0, padded_curves(curves[a], budget)).tolist() for a in order],
                keys=[labels[a] for a in order], title=f"{room} | IQM eval success rate", xname="Iteration"),
        })
        print(f"Logged {plot_path}")

def log_iqm_plots(sweep_dir: Path, architecture_dirs, early_stopping):
    """Log one IQM plot per room and one over all rooms, plus a table with their numbers."""
    rooms = sorted({d.parent.name for d in architecture_dirs})
    by_architecture = {}
    for d in architecture_dirs:
        by_architecture.setdefault(d.name, []).append(d)
    labels = iqm_labels(by_architecture)
    rng = np.random.default_rng(0)

    iqm_table = wandb.Table(columns=["room", "architecture", "label", "metric", "iqm", "ci_low", "ci_high"])
    for room in rooms + ["all_rooms"]:
        dirs = {architecture: [d for d in room_dirs if room in (d.parent.name, "all_rooms")]
                for architecture, room_dirs in by_architecture.items()}
        stats = iqm_stats({architecture: d for architecture, d in dirs.items() if d}, rng, early_stopping)

        fig = plot_iqm(stats, labels, title=room)
        plot_path = sweep_dir / f"iqm_{room}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        wandb.log({f"iqm/{room}": wandb.Image(str(plot_path), caption=room)})
        for architecture, metrics in stats.items():
            for key, (center, low, high) in metrics.items():
                iqm_table.add_data(room, architecture, labels[architecture], key, center, low, high)
        print(f"Logged {plot_path}")

    wandb.log({"iqm/summary": iqm_table})

@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    sweep_dir = Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp)
    architecture_dirs = sorted(p for p in sweep_dir.glob("*/*") if p.is_dir())
    if not architecture_dirs:
        raise FileNotFoundError(f"No <room>/<architecture> metric folders under {sweep_dir}")

    summary_columns = ["room", "architecture", "seeds", "auc_mean_of_means", "auc_ci95_halfwidth"] + [
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

            wandb.log({f"mean_of_means/{room}/{image_key(architecture)}": wandb.Image(str(plot_path), caption=architecture)})
            summary_table.add_data(room, architecture, ",".join(map(str, summary["seeds"])),
                                   *auc_mean_of_means(architecture_dir),
                                   *[summary[key][stat] for key, _, _ in MEAN_OF_MEANS_METRICS
                                     for stat in ("mean_of_means", "ci95_halfwidth")])
            print(f"Logged {plot_path}")

        wandb.log({"mean_of_means/summary": summary_table})
        log_iqm_plots(sweep_dir, architecture_dirs, cfg.algorithm.early_stopping)
        log_curve_plots(sweep_dir, architecture_dirs, cfg.algorithm.n_iterations, cfg.algorithm.eval_interval)


if __name__ == "__main__":
    main()
