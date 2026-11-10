"""
Seed-level statistics of a sweep, for the claims in the paper (plan: "Metrics and statistics").

The statistical unit is the training seed: every seed contributes one value per metric (its budget-padded
AUC, or the mean of its eval runs for success / SPL / nDTW), never its individual eval runs. Pooling the
10 x 10 eval episodes of each seed would inflate n and make small differences look significant.

For every <room>/<config>/ of <storage>/eval_metrics/<timestamp>/ it reports, per metric:
    - n seeds, mean, standard error, IQM with a 95% bootstrap CI over seeds (Agarwal et al. 2021);
and for every config against the reference config of its room (or every pair, without a reference):
    - difference of means, Hedges' g, Welch's t-test, Holm-corrected p over all comparisons of the metric,
      and the probability of improvement P(X > Y) with a bootstrap CI.
With several rooms it also reports each config's IQM over all (room x seed) runs with a stratified bootstrap.

    python scripts/seed_stats.py storage=<dir> timestamp=<sweep timestamp> wandb.group=<group>

Options (hydra overrides, all optional):
    +stats.vary=backbone +stats.reference=mlp   compare within every condition: configs that only differ in the
                                                varied field (bptt follows the backbone) against the one whose
                                                field equals the reference value, e.g. the memory gain per
                                                condition of E1 / E3
    +stats.reference=backbone:mlp               without vary: substring of the one reference config per room
                                                (no reference at all: every pair)
    +stats.metrics=[auc,success,spl,ndtw] +stats.alpha=0.01 +stats.n_boot=2000 +stats.seed=0
Results are printed, saved to <storage>/stats/<timestamp>.json and, unless wandb.mode=disabled, logged to W&B.
"""
import sys
import json
import warnings
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import numpy as np
from scipy.stats import ttest_ind
import wandb

from scripts.e4_promotion import parse_storage_name, iqm, stratified_bootstrap_ci

# metric -> (file in the seed folder, per-seed value from the saved array)
METRICS = {
    "auc":     ("auc.npy",         lambda values: float(np.ravel(values)[0])),
    "success": ("cr_values.npy",   lambda values: float(np.mean(values)) / 100.0),   # CR is stored in %
    "spl":     ("spl_values.npy",  lambda values: float(np.mean(values))),
    "ndtw":    ("ndtw_values.npy", lambda values: float(np.mean(values))),
}
SCREENING_SEEDS = {0, 1, 2}


def load_per_seed(sweep_dir: Path, metric: str):
    """{room: {config: {seed: value}}} for one metric; seeds without the metric's file are left out."""
    file_name, reduce = METRICS[metric]
    values = {}
    for path in sorted(sweep_dir.glob(f"*/*/seed_*/{file_name}")):
        values_array = np.load(path)
        if values_array.size == 0:     # e.g. nDTW of a seed that never reached the goal
            continue
        room, config, seed = path.parents[2].name, path.parents[1].name, int(path.parent.name.split("_")[-1])
        values.setdefault(room, {}).setdefault(config, {})[seed] = reduce(values_array)
    return values

def short_labels(configs):
    """Label every config by the storage-name fields that differ across the sweep."""
    fields = {config: parse_storage_name(config) for config in configs}
    keys = sorted({key for f in fields.values() for key in f})
    varying = [key for key in keys if len({f.get(key) for f in fields.values()}) > 1]
    return {config: " ".join(f"{key}:{fields[config].get(key, '-')}" for key in varying) or "config" for config in configs}

def comparison_pairs(names, reference, vary):
    """(config, reference config) pairs of one room, see the stats.vary / stats.reference options."""
    if vary is not None:
        # Condition = every field except the varied one (and bptt, which only exists for recurrent backbones,
        # and lr and the minibatch size, which follow the backbone)
        groups = {}
        for name in names:
            fields = parse_storage_name(name)
            condition = tuple(sorted((k, v) for k, v in fields.items() if k not in (vary, "bptt", "lr", "mb")))
            groups.setdefault(condition, []).append(name)
        pairs = []
        for members in groups.values():
            refs = [m for m in members if parse_storage_name(m).get(vary) == str(reference)]
            if len(refs) != 1:
                warnings.warn(f"No single {vary}:{reference} config in a condition of {len(members)} configs, skipped")
                continue
            pairs += [(m, refs[0]) for m in members if m != refs[0]]
        return pairs

    refs = [c for c in names if reference is not None and reference in c]
    if reference is not None and len(refs) != 1:
        warnings.warn(f"Reference '{reference}' matches {len(refs)} configs, comparing every pair")
        refs = []
    return ([(c, refs[0]) for c in names if c != refs[0]] if refs else
            [(a, b) for i, a in enumerate(names) for b in names[i + 1:]])

def bootstrap_ci(values, statistic, n_boot, rng):
    boots = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    correction = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(correction * (np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")

def probability_of_improvement(x, y):
    """P(X > Y) over all seed pairs, ties counted as 1/2 (Agarwal et al. 2021)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(np.mean((x > y) + 0.5 * (x == y)))

def holm(p_values):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    # Two zero-variance configs have a NaN p-value (e.g. CR 100 on every seed): no evidence of a difference
    p = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    stats = cfg.get("stats", {})
    reference = stats.get("reference", None)
    vary      = stats.get("vary", None)
    metrics   = list(stats.get("metrics", ["auc", "success", "spl", "ndtw"]))
    alpha     = float(stats.get("alpha", 0.01))
    n_boot    = int(stats.get("n_boot", 2000))
    seed      = int(stats.get("seed", 0))
    rng = np.random.default_rng(seed)

    sweep_dir = Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp)
    results = {"timestamp": str(cfg.timestamp), "reference": reference, "vary": vary, "alpha": alpha,
               "n_boot": n_boot, "seed": seed, "metrics": {}}

    for metric in metrics:
        per_room = load_per_seed(sweep_dir, metric)
        if not per_room:
            print(f"\n== {metric}: no {METRICS[metric][0]} under {sweep_dir}, skipped")
            continue
        configs = sorted({config for room in per_room.values() for config in room})
        labels = short_labels(configs)

        seeds_used = {s for room in per_room.values() for config in room.values() for s in config}
        if seeds_used & SCREENING_SEEDS:
            warnings.warn(f"Seeds {sorted(seeds_used & SCREENING_SEEDS)} are screening seeds: they may select configs "
                          f"but must not support reported claims (plan: 'Why these seed sets')")

        # -- Per room and config --
        summary = []
        for room, room_configs in sorted(per_room.items()):
            for config, by_seed in sorted(room_configs.items()):
                values = np.array(list(by_seed.values()))
                summary.append({
                    "room": room, "label": labels[config], "config": config, "seeds": sorted(by_seed),
                    "n": len(values), "mean": float(values.mean()),
                    "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan"),
                    "iqm": iqm(values), "iqm_ci": bootstrap_ci(values, iqm, n_boot, rng),
                })

        # -- Comparisons: config vs the room's reference, or every pair --
        comparisons = []
        for room, room_configs in sorted(per_room.items()):
            for a, b in comparison_pairs(sorted(room_configs), reference, vary):
                x, y = np.array(list(room_configs[a].values())), np.array(list(room_configs[b].values()))
                if len(x) < 2 or len(y) < 2:
                    continue
                p_value = float(ttest_ind(x, y, equal_var=False).pvalue)
                boot_poi = [probability_of_improvement(rng.choice(x, len(x)), rng.choice(y, len(y)))
                            for _ in range(n_boot)]
                comparisons.append({
                    "room": room, "a": labels[a], "b": labels[b], "n_a": len(x), "n_b": len(y),
                    "diff": float(x.mean() - y.mean()), "hedges_g": hedges_g(x, y), "p_welch": p_value,
                    "p_of_improvement": probability_of_improvement(x, y),
                    "poi_ci": (float(np.percentile(boot_poi, 2.5)), float(np.percentile(boot_poi, 97.5))),
                })
        # Holm over every comparison of this metric, the family the paper's claims are drawn from
        for comparison, adjusted in zip(comparisons, holm([c["p_welch"] for c in comparisons])):
            comparison["p_holm"] = adjusted
            comparison["significant"] = bool(adjusted < alpha)

        # -- Across rooms: IQM over every (room x seed) run of a config, seeds resampled within rooms --
        aggregate = []
        if len(per_room) > 1:
            by_label = {}
            for room, room_configs in per_room.items():
                for config, by_seed in room_configs.items():
                    by_label.setdefault(labels[config], []).append(np.array(list(by_seed.values())))
            for label, per_room_scores in sorted(by_label.items()):
                aggregate.append({"label": label, "rooms": len(per_room_scores),
                                  "iqm": iqm(np.concatenate(per_room_scores)),
                                  "iqm_ci": stratified_bootstrap_ci(per_room_scores, n_boot, rng)})

        results["metrics"][metric] = {"summary": summary, "comparisons": comparisons, "aggregate": aggregate}

        print(f"\n== {metric} (unit: training seed)")
        for row in summary:
            print(f"  {row['room']:16s} {row['label']:44s} n={row['n']:2d} mean={row['mean']:.3f} ± {row['se']:.3f} SE  "
                  f"IQM={row['iqm']:.3f} [{row['iqm_ci'][0]:.3f}, {row['iqm_ci'][1]:.3f}]")
        for c in comparisons:
            print(f"  {c['room']:16s} {c['a']} vs {c['b']}: diff={c['diff']:+.3f} g={c['hedges_g']:+.2f} "
                  f"p={c['p_welch']:.3g} p_holm={c['p_holm']:.3g} P(a>b)={c['p_of_improvement']:.2f} "
                  f"[{c['poi_ci'][0]:.2f}, {c['poi_ci'][1]:.2f}] -> {'significant' if c['significant'] else 'n.s.'} at {alpha}")
        for row in aggregate:
            print(f"  all rooms        {row['label']:44s} IQM={row['iqm']:.3f} [{row['iqm_ci'][0]:.3f}, {row['iqm_ci'][1]:.3f}]")

    out_path = Path(cfg.storage) / "stats" / f"{cfg.timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} (bootstrap seed {seed})")

    if cfg.wandb.mode != "disabled":
        table = wandb.Table(columns=["metric", "room", "a", "b", "n_a", "n_b", "diff", "hedges_g",
                                     "p_welch", "p_holm", "p_of_improvement", "significant"])
        for metric, block in results["metrics"].items():
            for c in block["comparisons"]:
                table.add_data(metric, c["room"], c["a"], c["b"], c["n_a"], c["n_b"], c["diff"], c["hedges_g"],
                               c["p_welch"], c["p_holm"], c["p_of_improvement"], c["significant"])
        with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, group=cfg.wandb.group,
                        name=f"seed_stats_{cfg.timestamp}", job_type="eval",
                        config={"storage": str(cfg.storage), "timestamp": str(cfg.timestamp), "alpha": alpha,
                                "reference": reference, "n_boot": n_boot, "seed": seed},
                        tags=["eval", "seed_stats", f"timestamp:{cfg.timestamp}"], mode=cfg.wandb.mode):
            wandb.log({"seed_stats/comparisons": table})


if __name__ == "__main__":
    main()
