"""
Promotion analysis between the stages of the E4 staged ablation.

Reads the budget-padded AUC (auc.npy) of every seed in <storage>/eval_metrics/<timestamp>/<room>/<config>/,
scores each config by the IQM over all its runs (seeds x rooms) with a 95% stratified-bootstrap CI
(seeds resampled within each room, Agarwal et al. 2021), and applies the plan's promotion rule per theme:

    keep = theme leader + every config whose CI overlaps the leader's (at most ceil(n/3) of the n levels)
         + one random wildcard among the rest + the baseline (always)

Configs are assigned to themes by the fields of their storage name that differ from the baseline. Each
theme only varies its own fields, so the baseline's value of every field is the most common one. From Stage 2
the aux configs and Stage 1's baseline stay on Stage 1's encoder and reward, so pass the re-baselined fields:
+promotion.base.encoder=cnn +promotion.base.cpen=0.01

Aux-on levels run with distances masked (the depth head needs a hidden target), so they are ranked against
their matched reference, the same config with the aux head off (e.g. the masked LSTM for the aux theme),
not against the unmasked baseline. Aux-on levels without a matched reference in the sweep fall back to the
baseline in a separate "<theme> unmatched" block, whose ranking mixes the aux effect with the masking.
Likewise a recurrent config with a BPTT window other than the sweep's usual one (the most common) forms the
"bptt" theme, ranked against the same config at the usual window.

    python scripts/e4_promotion.py storage=<dir> timestamp=<sweep timestamp> wandb.group=e4_stage1

Options (hydra overrides, all optional): +promotion.n_boot=2000 +promotion.seed=0 (bootstrap and
wildcard RNG, report it). Results are printed, saved to <storage>/promotion/<timestamp>.json and,
unless wandb.mode=disabled, logged as a table to W&B.
"""
import sys
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import numpy as np
from scipy.stats import trim_mean
import wandb

# Fields of the storage name (architecture name + bptt + variant_name in train.py), in order.
# Splitting on these keys, not on "_", keeps values like room_local and frame_stack intact
STORAGE_KEYS = ["encoder", "backbone", "frame_stack", "grid_cell", "auxiliary_head", "bptt", "colour", "goal",
                "ginfo", "pinfo", "ggrid", "fov", "act", "cpen", "gamma", "ep", "mask", "width", "lr", "aux",
                "depth", "modact", "mb", "bs"]
_KEY_SPLIT = re.compile(r"_(?=(?:" + "|".join(STORAGE_KEYS) + r"):)")

# Theme of a config = the fields it changes; a field set maps to the first theme that contains it.
# The aux theme runs on the LSTM, so a different backbone marks it. Aux on the base backbone belongs to
# capacity (its matched-width points, including width 64 = the base width)
THEME_FIELDS = [
    ("aux",         {"backbone"}),
    ("capacity",    {"width", "aux", "depth", "modact"}),
    ("observation", {"encoder", "fov"}),
    ("action",      {"act"}),
    ("reward",      {"cpen", "gamma"}),
]


def parse_storage_name(name: str):
    """'encoder:mlp_backbone:lstm_..._lr:0.0003' -> {'encoder': 'mlp', 'backbone': 'lstm', ..., 'lr': '0.0003'}"""
    return dict(field.split(":", 1) for field in _KEY_SPLIT.split(name))

def theme_of(changed: set):
    if not changed:
        return "baseline"
    for theme, fields in THEME_FIELDS:
        if fields & changed:
            return theme
    return "other"

def iqm(scores):
    """Interquartile mean: mean of the middle 50% of the runs."""
    return float(trim_mean(np.asarray(scores), proportiontocut=0.25))

def stratified_bootstrap_ci(per_room, n_boot, rng):
    """
    95% percentile CI of the IQM over all runs, resampling seeds with replacement within each room,
    so every bootstrap sample keeps the same number of runs per room.
    per_room: list of 1-D arrays, the per-seed scores of one room each
    """
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = iqm(np.concatenate([rng.choice(scores, size=len(scores), replace=True) for scores in per_room]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def reference_fields(config_fields, base_bptt):
    """
    Fields of a level's matched reference, or None if its reference is the baseline:
    aux on -> the same config with the aux head off (same backbone, masking, width, ...),
    a BPTT window other than the sweep's usual one -> the same config at the usual window
    """
    if config_fields.get("auxiliary_head") == "True":
        target = {key: value for key, value in config_fields.items() if key != "aux"}
        target["auxiliary_head"] = "False"
        return target
    if config_fields.get("bptt", base_bptt) != base_bptt:
        # Windows at equal updates change the minibatch with the window, the usual window keeps the default
        return {key: value for key, value in dict(config_fields, bptt=base_bptt).items() if key != "mb"}
    return None

def matched_reference(target, fields):
    """Config of the sweep with exactly these fields, or None. fields: {config: parsed storage name} of the whole sweep"""
    return next((config for config, f in fields.items() if f == target), None)

def promote(levels, rng, reference="baseline"):
    """
    The plan's rule for one theme. levels: list of dicts with "label", "iqm", "ci" (the reference included).
    reference: label of the level that is always kept. Returns {label: reason} for every kept level.
    """
    ranked = sorted(levels, key=lambda level: level["iqm"], reverse=True)
    leader = ranked[0]
    cap = math.ceil(len(levels) / 3)
    kept = {leader["label"]: "leader"}
    for level in ranked[1:]:
        if len(kept) >= cap:
            break
        if level["ci"][1] >= leader["ci"][0]:   # CIs overlap
            kept[level["label"]] = "CI overlaps leader"
    rest = [level["label"] for level in ranked if level["label"] not in kept and level["label"] != reference]
    if rest:
        kept[rest[rng.integers(len(rest))]] = "wildcard"
    kept.setdefault(reference, f"{'baseline' if reference == 'baseline' else 'reference'} (always kept)")
    return kept

def print_block(block, levels, kept, rooms, reference):
    print(f"\n== {block} ({len(levels)} levels, keep at most {math.ceil(len(levels) / 3)} + wildcard + {reference})")
    for level in sorted(levels, key=lambda level: level["iqm"], reverse=True):
        per_room = "  ".join(f"{room}={level['mean_per_room'].get(room, float('nan')):.3f}" for room in rooms)
        runs = sum(level["runs"].values())
        print(f"  {level['label']:42s} IQM={level['iqm']:.3f} [{level['ci'][0]:.3f}, {level['ci'][1]:.3f}] "
              f"runs={runs:2d}  {per_room}  -> {kept.get(level['label'], 'dropped')}")


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    promotion = cfg.get("promotion", {})
    n_boot = int(promotion.get("n_boot", 2000))
    seed   = int(promotion.get("seed", 0))
    rng = np.random.default_rng(seed)

    sweep_dir = Path(cfg.storage) / "eval_metrics" / str(cfg.timestamp)
    # config storage name -> room -> per-seed AUCs
    aucs = {}
    for auc_path in sorted(sweep_dir.glob("*/*/seed_*/auc.npy")):
        room, config = auc_path.parents[2].name, auc_path.parents[1].name
        aucs.setdefault(config, {}).setdefault(room, []).append(float(np.load(auc_path)[0]))
    if not aucs:
        raise FileNotFoundError(f"No <room>/<config>/seed_*/auc.npy under {sweep_dir}")

    fields = {config: parse_storage_name(config) for config in aucs}
    keys = sorted({key for f in fields.values() for key in f})
    base_fields = {key: Counter(f.get(key) for f in fields.values()).most_common(1)[0][0] for key in keys}
    base_fields.update({key: str(value) for key, value in promotion.get("base", {}).items()})
    rooms = sorted({room for per_room in aucs.values() for room in per_room})
    # The usual BPTT window of the recurrent configs; the others are levels of the bptt theme
    windows = Counter(f["bptt"] for f in fields.values() if "bptt" in f)
    base_bptt = windows.most_common(1)[0][0] if windows else None

    configs = []
    for config, per_room in aucs.items():
        changed = {key for key in keys if fields[config].get(key) != base_fields[key]}
        # The label only shows the fields that define the level, not the ones that follow from them
        # (the masking follows from an aux head, but defines the level of an aux-off reference; the
        # usual BPTT window follows from a recurrent backbone, any other window defines a bptt level)
        other_window = fields[config].get("bptt", base_bptt) != base_bptt
        implied = ({"lr", "auxiliary_head"} | ({"bptt"} if not other_window else set())
                   | ({"mask"} if fields[config].get("auxiliary_head") == "True" else set()))
        # The aux configs (recurrent or masked) keep Stage 1's encoder and reward at every stage
        if {"backbone", "mask"} & changed:
            implied |= {"encoder", "cpen"}
        shown = sorted(changed - implied) or sorted(changed)
        # A plain recurrent config (no masking, no aux head) is a memory level, not an aux one. On the base
        # encoder any recurrent label has to show the encoder, or it reads the same as the LSTM on Stage 1's encoder
        memory = "backbone" in changed and not {"mask", "auxiliary_head"} & changed
        if "backbone" in changed and "encoder" not in changed:
            shown = sorted(set(shown) | {"encoder"})
        label = " ".join(f"{key}:{fields[config].get(key)}" for key in shown) or "baseline"
        per_room_scores = [np.array(per_room[room]) for room in rooms if room in per_room]
        all_scores = np.concatenate(per_room_scores)
        configs.append({
            "label": label, "theme": "bptt" if other_window else "memory" if memory else theme_of(changed), "config": config,
            "runs": {room: len(per_room.get(room, [])) for room in rooms},
            "iqm": iqm(all_scores), "ci": stratified_bootstrap_ci(per_room_scores, n_boot, rng),
            "mean_per_room": {room: float(np.mean(per_room[room])) for room in rooms if room in per_room},
        })

    baseline = [c for c in configs if c["theme"] == "baseline"]
    if len(baseline) != 1:
        raise RuntimeError(f"Expected one baseline config, found {len(baseline)}: {[c['config'] for c in baseline]}")

    # A config missing runs pools fewer rooms, so its IQM leans towards the rooms it has (interim analyses)
    most_runs = {room: max(c["runs"][room] for c in configs) for room in rooms}
    for c in configs:
        missing = {room: most_runs[room] - n for room, n in c["runs"].items() if n < most_runs[room]}
        if missing:
            print(f"Warning: {c['label']} is missing runs {missing}, its IQM is not comparable yet")

    # Aux-on levels are ranked against their matched aux-off config (same masking), BPTT levels against
    # the usual window, every other level against the baseline. A config that is some level's matched
    # reference anchors that block instead
    by_config = {c["config"]: c for c in configs}
    for c in configs:
        target = reference_fields(fields[c["config"]], base_bptt)
        c["reference"] = baseline[0]["config"] if target is None else matched_reference(target, fields)
    references = {c["reference"] for c in configs} - {None, baseline[0]["config"]}

    results = {"timestamp": str(cfg.timestamp), "rooms": rooms, "n_boot": n_boot, "seed": seed,
               "base_fields": base_fields, "base_bptt": base_bptt, "themes": {}}
    for theme in [t for t, _ in THEME_FIELDS] + ["bptt", "memory", "other"]:
        # CNN + LSTM anchors its aux block, but it's still ranked against the baseline as a memory level
        theme_levels = [c for c in configs if c["theme"] == theme and (c["config"] not in references or (
            theme == "memory" and fields[c["config"]].get("encoder") == base_fields["encoder"]))]
        for reference in dict.fromkeys(c["reference"] for c in theme_levels):
            levels = [c for c in theme_levels if c["reference"] == reference]
            if reference is None:
                # No matched reference at this point of the sweep: rank against the baseline, but keep it apart
                block, anchor = f"{theme} unmatched", baseline[0]
            else:
                anchor = by_config[reference]
                block = theme if anchor is baseline[0] else f"{theme} vs {anchor['label']}"
            levels = levels + [anchor]
            kept = promote(levels, rng, reference=anchor["label"])
            results["themes"][block] = {"reference": anchor["label"],
                                        "levels": [dict(level, kept=kept.get(level["label"])) for level in levels]}
            print_block(block, levels, kept, rooms, anchor["label"])
            if reference is None:
                print("  (no matched reference in the sweep: this ranking mixes the level's effect with how it differs from the baseline)")

    out_path = Path(cfg.storage) / "promotion" / f"{cfg.timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} (bootstrap/wildcard seed {seed})")

    if cfg.wandb.mode != "disabled":
        table = wandb.Table(columns=["theme", "level", "iqm_auc", "ci_low", "ci_high", "runs", "decision"]
                                    + [f"mean_auc_{room}" for room in rooms] + ["config"])
        for theme, block in results["themes"].items():
            for level in block["levels"]:
                table.add_data(theme, level["label"], level["iqm"], level["ci"][0], level["ci"][1],
                               sum(level["runs"].values()), level["kept"] or "dropped",
                               *[level["mean_per_room"].get(room, float("nan")) for room in rooms], level["config"])
        with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, group=cfg.wandb.group,
                        name=f"promotion_{cfg.timestamp}", job_type="eval",
                        config={"storage": str(cfg.storage), "timestamp": str(cfg.timestamp), "n_boot": n_boot, "seed": seed},
                        tags=["eval", "promotion", f"timestamp:{cfg.timestamp}"], mode=cfg.wandb.mode):
            wandb.log({"promotion/summary": table})


if __name__ == "__main__":
    main()
