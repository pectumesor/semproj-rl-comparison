"""
Figures and tables of the paper (plan: "Figures"), drawn from the saved seed folders of a finished sweep.

Every figure is saved as .pdf (for the paper) and .png under <storage>/figures/<timestamp>/, with a .csv of
the numbers it shows. The unit is the training seed: IQM over seeds with a 95% bootstrap CI over seeds, as in
scripts/seed_stats.py.

    python scripts/paper_figures.py +figure.name=<name> storage=<dir> timestamp=<sweep timestamp> wandb.group=<group>

    name   sweep                 output
    e1     e1_place_code_audit   fig2_left: AUC per colour/goal condition and room, memory gain vs aliasing score
                                 (scripts/aliasing_score.py, <storage>/aliasing/scores.jsonl)
    e1b    e1b_fov               fig_e1b_fov: AUC vs field of view per backbone
    e2     e2_twin_rooms         fig3: success vs L per backbone, BPTT window x L heatmap;
                                 table_e2: the checks of scripts/e2_diagnostics.py, as .csv (all L) and .tex (one L)
    e3     e3_information        fig4: AUC per information condition and backbone, memory gain row
    e4     e4_stage<n>           fig2_right: AUC effect of every level against its reference, per room, and the
                                 Kendall tau between rooms. Run scripts/e4_promotion.py first

The memory gain is always LSTM - MLP, never "best of LSTM and GRU" (plan: GRU trigger rule). The stars of the
gain row are Welch's t, Holm-corrected over the row's whole figure, at +figure.alpha.

Options (hydra overrides, all optional):
    +figure.n_boot=2000 +figure.seed=0 +figure.alpha=0.01
    +figure.window=64        e2: BPTT window of the LSTM and GRU lines (the heatmap shows every LSTM window)
    +figure.table_L=128      e2: memory horizon of the .tex table
    +figure.promotion=<json> e4: promotion analysis to use, default <storage>/promotion/<timestamp>.json
    +figure.compare=<json>   e4: promotion analysis of another stage, adds the tau between the stages per level name
Unless wandb.mode=disabled the figure is also logged to W&B.
"""
import sys
import csv
import json
import logging
import warnings
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import hydra
from omegaconf import DictConfig
import matplotlib
matplotlib.use("Agg")  # No display on the cluster
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import kendalltau, spearmanr, ttest_ind
import wandb

from scripts.e4_promotion import parse_storage_name, iqm
from scripts.seed_stats import load_per_seed, bootstrap_ci, holm, SCREENING_SEEDS

# Column widths of a two-column paper, in inches
SINGLE_COLUMN, DOUBLE_COLUMN = 3.3, 6.9

INK, INK_SECONDARY = "#0b0b0b", "#52514e"
GRID, AXIS, MISSING = "#e1e0d9", "#c3c2b7", "#f0efec"
LINE_WIDTH, MARKER_SIZE = 1.5, 5

# A backbone keeps its colour and marker in every figure, the marker also carries it in greyscale prints.
# Colour order checked with the dataviz validator (adjacent pairs, CVD safe)
SERIES = {"MLP": ("#2a78d6", "o"), "LSTM": ("#eb6834", "s"), "GRU": ("#1baf7a", "^"),
          "FS4": ("#eda100", "D"), "FS16": ("#e87ba4", "v")}
BACKBONES = {"mlp": "MLP", "lstm": "LSTM", "gru": "GRU"}
# Row order of the E2 results: memoryless, then by how much the backbone can remember
E2_ORDER = ["MLP", "FS4", "FS16", "LSTM", "GRU"]

SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
SEQUENTIAL.set_bad(MISSING)
# Blue = the MLP is better, orange = the LSTM is better, the colours of the two backbones
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#2a78d6", MISSING, "#eb6834"])
DIVERGING.set_bad(MISSING)

ROOM_LABELS = {"two_room": "Two rooms", "four_room": "Four rooms", "C_maze": "C maze",
               "medium_maze": "Medium maze", "hard_maze": "Hard maze"}
ROOM_ORDER = list(ROOM_LABELS)

# (colour mode, goal mode, tick label) of the E1 conditions
E1_CONDITIONS = [("global", "fixed", "Perlin\nfixed"), ("off", "fixed", "Off\nfixed"), ("off", "random", "Off\nrand.")]
# Storage fields (goal info, position info, goal grid) -> the plan's name of the E3 condition, in the plan's order
E3_CONDITIONS = {
    ("none", "none", "False"):              "I0",
    ("oracle", "none", "False"):            "G1",
    ("last_seen_updated", "none", "False"): "G2a",
    ("last_seen_frozen", "none", "False"):  "G2b",
    ("none", "xy", "False"):                "P1",
    ("none", "rff", "False"):               "P2",
    ("none", "grid", "False"):              "P3",
    ("none", "noisy_grid", "False"):        "P4",
    ("none", "grid", "True"):               "P5",
}


def paper_style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.edgecolor": AXIS,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 300,
        "pdf.fonttype": 42,     # embedded TrueType fonts, submission systems reject Type 3
    })

def ordered_rooms(rooms):
    return sorted(rooms, key=lambda room: ROOM_ORDER.index(room) if room in ROOM_ORDER else len(ROOM_ORDER))

def collect(sweep_dir: Path, metric: str, key_of):
    """
    {(room, key): per-seed values} of a sweep. key_of(fields of the storage name) names a config inside its room,
    None leaves it out. Two configs with the same key would silently replace each other, so that is an error.
    """
    per_room = load_per_seed(sweep_dir, metric)
    if not per_room:
        raise FileNotFoundError(f"No {metric} results under {sweep_dir}")
    cells = {}
    for room, configs in per_room.items():
        for config, by_seed in configs.items():
            key = key_of(parse_storage_name(config))
            if key is None:
                continue
            if (room, key) in cells:
                raise RuntimeError(f"Two configs of {room} are the cell {key}, the last one is {config}")
            cells[room, key] = by_seed
    if not cells:
        raise FileNotFoundError(f"No config of {sweep_dir} belongs to this figure")

    seeds_used = {seed for by_seed in cells.values() for seed in by_seed}
    if seeds_used & SCREENING_SEEDS:
        warnings.warn(f"Seeds {sorted(seeds_used & SCREENING_SEEDS)} are screening seeds, they must not support "
                      f"reported claims (plan: 'Why these seed sets')")
    usual = Counter(len(by_seed) for by_seed in cells.values()).most_common(1)[0][0]
    for cell, by_seed in cells.items():
        if len(by_seed) != usual:
            warnings.warn(f"{cell} has {len(by_seed)} seeds, the others {usual}: a run failed or is still going")
    return {cell: np.array(list(by_seed.values())) for cell, by_seed in cells.items()}

def score(values, rng, n_boot):
    low, high = bootstrap_ci(values, iqm, n_boot, rng) if len(values) > 1 else (float("nan"), float("nan"))
    return {"n": len(values), "iqm": iqm(values), "ci_low": low, "ci_high": high}

def iqm_difference(x, y, rng, n_boot):
    """IQM(x) - IQM(y) with a 95% bootstrap CI, seeds resampled within each config."""
    boots = [iqm(rng.choice(x, len(x))) - iqm(rng.choice(y, len(y))) for _ in range(n_boot)]
    return iqm(x) - iqm(y), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def dot_with_ci(ax, x, row, series, label=None):
    colour, marker = SERIES[series]
    ax.plot([x, x], [row["ci_low"], row["ci_high"]], color=colour, lw=LINE_WIDTH, solid_capstyle="round")
    ax.plot(x, row["iqm"], marker, color=colour, ms=MARKER_SIZE, mec="white", mew=0.6, ls="none", label=label)

def heatmap(ax, values, cmap, norm, row_labels, column_labels, marks=None):
    """Cells with their value printed, so no colour bar is needed. values: (rows, columns), NaN = no runs."""
    values = np.asarray(values, dtype=float)
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, norm=norm, edgecolors="white", linewidth=1.5)
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        # Signed values drop the leading zero, the star has to fit in the cell too
        text = "–" if np.isnan(value) else f"{value:+.2f}".replace("0.", ".", 1) if norm.vmin < 0 else f"{value:.2f}"
        if marks is not None and marks[row][column]:
            text += "*"
        red, green, blue, _ = cmap(norm(value)) if not np.isnan(value) else (1.0, 1.0, 1.0, 1.0)
        dark_cell = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.62
        ax.text(column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=6,
                color="white" if dark_cell else INK)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, column_labels)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, row_labels)
    ax.set_ylim(values.shape[0], 0)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)


# -- E1: does the MLP's lead come from the place code? --

def aliasing_scores(storage: Path, fov):
    """{(room, colour mode): score} of scripts/aliasing_score.py, the file is appended to so the last run wins."""
    path = storage / "aliasing" / "scores.jsonl"
    if not path.exists():
        warnings.warn(f"No {path}, leaving out the memory gain vs aliasing panel")
        return {}
    scores = {}
    with open(path) as f:
        for line in f:
            result = json.loads(line)
            if result["fov"] == fov:
                scores[result["room"], result["colour_mode"]] = result["score"]
    return scores

def figure_e1(sweep_dir: Path, storage: Path, figure, rng, cfg):
    n_boot = int(figure.get("n_boot", 2000))
    cells = collect(sweep_dir, "auc", lambda f: (f["colour"], f["goal"], f["backbone"]))
    scores = aliasing_scores(storage, cfg.env.fov)
    rooms = ordered_rooms({room for room, _ in cells})
    backbones = [b for b in BACKBONES if any(key[2] == b for _, key in cells)]

    fig, axes = plt.subplots(1, len(rooms) + bool(scores), figsize=(DOUBLE_COLUMN, 2.2), layout="constrained",
                             width_ratios=[1] * len(rooms) + [1.5] * bool(scores))
    axes = np.atleast_1d(axes)
    rows, points = [], []
    dodge = np.linspace(-0.15, 0.15, len(backbones)) if len(backbones) > 1 else [0.0]
    for ax, room in zip(axes, rooms):
        for x, (colour, goal, _) in enumerate(E1_CONDITIONS):
            for shift, backbone in zip(dodge, backbones):
                if (room, (colour, goal, backbone)) not in cells:
                    continue
                row = score(cells[room, (colour, goal, backbone)], rng, n_boot)
                rows.append({"room": room, "colour": colour, "goal": goal, "backbone": backbone, **row})
                dot_with_ci(ax, x + shift, row, BACKBONES[backbone],
                            label=BACKBONES[backbone] if x == 0 and ax is axes[0] else None)
            # The aliasing score only depends on what the agent sees, so it pairs with the fixed-goal conditions
            pair = (room, (colour, goal, "lstm")), (room, (colour, goal, "mlp"))
            if goal == "fixed" and (room, colour) in scores and all(cell in cells for cell in pair):
                points.append({"room": room, "colour": colour, "aliasing": scores[room, colour],
                               "memory_gain": iqm(cells[pair[0]]) - iqm(cells[pair[1]])})
        ax.set_xticks(range(len(E1_CONDITIONS)), [label for _, _, label in E1_CONDITIONS], fontsize=6)
        ax.set_xlim(-0.5, len(E1_CONDITIONS) - 0.5)
        ax.set_ylim(0.0, 1.03)
        ax.set_title(ROOM_LABELS.get(room, room))
        ax.grid(axis="x", visible=False)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Success AUC (IQM)")
    handles, labels = axes[0].get_legend_handles_labels()

    if scores:
        ax = axes[-1]
        ax.axhline(0.0, color=AXIS, lw=0.8)
        for point in points:
            filled = point["colour"] != "off"
            ax.plot(point["aliasing"], point["memory_gain"], "o", ms=MARKER_SIZE, color=INK_SECONDARY,
                    mfc=INK_SECONDARY if filled else "white", ls="none")
            ax.annotate(ROOM_LABELS.get(point["room"], point["room"]), (point["aliasing"], point["memory_gain"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=6, color=INK_SECONDARY)
        ax.margins(x=0.25, y=0.2)
        handles += [Line2D([], [], marker="o", ms=MARKER_SIZE, color=INK_SECONDARY, ls="none"),
                    Line2D([], [], marker="o", ms=MARKER_SIZE, color=INK_SECONDARY, mfc="white", ls="none")]
        labels += ["Perlin colour", "Colour off"]
        ax.set_xlabel("Aliasing score")
        ax.set_ylabel("Memory gain (LSTM − MLP)")
        if len(points) >= 3:
            rho = spearmanr([p["aliasing"] for p in points], [p["memory_gain"] for p in points]).statistic
            ax.set_title(f"Spearman ρ = {rho:.2f}, n = {len(points)}")
    fig.legend(handles, labels, loc="outside upper center", ncols=len(handles))
    return "fig2_left", fig, rows + points


# -- E1b: narrowing the view should increase the memory gain --

def figure_e1b(sweep_dir: Path, storage: Path, figure, rng, cfg):
    n_boot = int(figure.get("n_boot", 2000))
    cells = collect(sweep_dir, "auc", lambda f: (int(f["fov"]), f["backbone"]))
    fovs = sorted({key[0] for _, key in cells})
    backbones = [b for b in BACKBONES if any(key[1] == b for _, key in cells)]

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.2), layout="constrained")
    rows = []
    for backbone in backbones:
        colour, marker = SERIES[BACKBONES[backbone]]
        line = []
        for room, (fov, b) in cells:
            if b == backbone:
                line.append({"room": room, "fov": fov, "backbone": backbone, **score(cells[room, (fov, b)], rng, n_boot)})
        line.sort(key=lambda row: row["fov"])
        rows += line
        x = [row["fov"] for row in line]
        ax.fill_between(x, [row["ci_low"] for row in line], [row["ci_high"] for row in line], color=colour,
                        alpha=0.15, lw=0)
        ax.plot(x, [row["iqm"] for row in line], color=colour, marker=marker, ms=MARKER_SIZE, mec="white", mew=0.6,
                lw=LINE_WIDTH, label=BACKBONES[backbone])
        # At the narrow end: the lines are expected to meet at the wide one, the labels would collide there
        ax.annotate(BACKBONES[backbone], (x[0], line[0]["iqm"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(fovs, [f"{fov}°" for fov in fovs])
    ax.set_xlabel("Field of view")
    ax.set_ylabel("Success AUC (IQM)")
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower right", ncols=len(backbones))
    return "fig_e1b_fov", fig, rows


# -- E2: success vs memory horizon --

def e2_key(fields):
    """(series, BPTT window): frame stacks and the MLP have no window."""
    if fields["backbone"] == "mlp":
        stack = fields["frame_stack"]
        return ("MLP" if stack == "False" else f"FS{stack.removeprefix('True')}"), None
    return BACKBONES[fields["backbone"]], int(fields["bptt"])

def memory_horizon(room: str):
    return int(room.rsplit("_L", 1)[1])

def e2_order(L, series, window):
    return L, E2_ORDER.index(series), window or 0

def e2_table(sweep_dir: Path, out_dir: Path, table_L: int):
    """
    The memory checks of scripts/e2_diagnostics.py: mean and standard error over seeds per L and config.
    Saved as table_e2.csv (every L) and table_e2.tex (table_L only, probe at the bin that holds L, i.e. how
    well the cue can still be read from the hidden state when the agent reaches the junction).
    """
    by_config = {}
    for path in sorted(sweep_dir.glob("*/*/seed_*/e2_diagnostics.json")):
        with open(path) as f:
            result = json.load(f)
        L = memory_horizon(path.parents[2].name)
        by_config.setdefault((L, *e2_key(parse_storage_name(path.parents[1].name))), []).append(result)
    if not by_config:
        warnings.warn(f"No e2_diagnostics.json under {sweep_dir}, no table")
        return

    def mean_se(values):
        values = [value for value in values if value is not None]
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")

    def correct_room(result):
        """Share of the cued room among the episodes that chose a room, timeouts left out."""
        confusion = np.array(result["memory"]["confusion"])[:, :-1]
        return float(np.trace(confusion) / confusion.sum()) if confusion.sum() > 0 else None

    rows = []
    for (L, series, window), results in sorted(by_config.items(), key=lambda item: e2_order(*item[0])):
        row = {"L": L, "series": series, "window": window, "seeds": len(results)}
        columns = {"success":     [r["memory"]["success"] for r in results],
                   "wrong_room":  [r["memory"]["wrong_room"] for r in results],
                   "timeout":     [r["memory"]["timeout"] for r in results],
                   "correct_room": [correct_room(r) for r in results],
                   "reached_junction": [r["memory"].get("reached_junction") for r in results],
                   "reset_success": [r.get("reset_at_junction", {}).get("success") for r in results],
                   "reset_drop":  [r.get("reset_drop") for r in results],
                   # Body vs recurrent memory, over the episodes that got to the intervention point
                   "mirror_baseline": [r.get("mirror", {}).get("baseline", {}).get("success") for r in results],
                   "mirror_success": [r.get("mirror", {}).get("mirrored", {}).get("success") for r in results]}
        for name in ("hidden_other", "pose_other", "both_other", "hidden_same"):
            columns[f"swap_{name}"] = [r.get("swap", {}).get(name, {}).get("success") for r in results]
        for sweep in results[0].get("reset_sweep", []):
            fraction = sweep["corridor_fraction"]
            columns[f"reset_at_{fraction:g}"] = [next((row.get("success") for row in r.get("reset_sweep", [])
                                                       if row["corridor_fraction"] == fraction), None) for r in results]
        # Bins as in PROBE_BINS, a seed without a fitted probe in a bin has accuracy None
        for low, high in [probe["bin"] for probe in results[0].get("probe", [])]:
            columns[f"probe_{low}_{high}"] = [next(p["accuracy"] for p in r["probe"] if p["bin"] == [low, high])
                                              for r in results]
        for name, values in columns.items():
            row[name], row[f"{name}_se"] = mean_se(values)
        rows.append(row)
    save_rows(rows, out_dir / "table_e2.csv")

    def cell(row, name):
        return "--" if np.isnan(row.get(name, float("nan"))) else f"{row[name]:.2f} $\\pm$ {row[f'{name}_se']:.2f}"

    def probe_cell(row):
        for name in row:
            parts = name.split("_")
            if name.startswith("probe_") and len(parts) == 3 and int(parts[1]) <= row["L"] <= int(parts[2]):
                return cell(row, name)
        return "--"

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Backbone & Success & Correct room & After reset & Drop & Cue probe \\", r"\midrule"]
    for row in (row for row in rows if row["L"] == table_L):
        name = row["series"] + (f"-{row['window']}" if row["window"] else "")
        lines.append(f"{name} & {cell(row, 'success')} & {cell(row, 'correct_room')} & {cell(row, 'reset_success')} & "
                     f"{cell(row, 'reset_drop')} & {probe_cell(row)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out_dir / "table_e2.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out_dir / 'table_e2.csv'} and table_e2.tex (L={table_L})")

def figure_e2(sweep_dir: Path, storage: Path, figure, rng, cfg):
    n_boot = int(figure.get("n_boot", 2000))
    window = int(figure.get("window", 64))
    cells = collect(sweep_dir, "success", e2_key)
    cells = {(memory_horizon(room), key): values for (room, key), values in cells.items()}
    horizons = sorted({L for L, _ in cells})
    rows = [{"L": L, "series": key[0], "window": key[1], **score(values, rng, n_boot)}
            for (L, key), values in sorted(cells.items(), key=lambda item: e2_order(item[0][0], *item[0][1]))]

    fig, (ax, ax_windows) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.5), layout="constrained", width_ratios=[1.8, 1])

    for series in E2_ORDER:
        recurrent = series in ("LSTM", "GRU")
        line = [row for row in rows if row["series"] == series and (row["window"] == window if recurrent else True)]
        if not line:
            continue
        colour, marker = SERIES[series]
        x = [row["L"] for row in line]
        ax.fill_between(x, [row["ci_low"] for row in line], [row["ci_high"] for row in line], color=colour,
                        alpha=0.15, lw=0)
        ax.plot(x, [row["iqm"] for row in line], color=colour, marker=marker, ms=MARKER_SIZE, mec="white", mew=0.6,
                lw=LINE_WIDTH, label=f"{series} (BPTT {window})" if recurrent else series)

    ax.axhline(0.5, color=INK_SECONDARY, lw=0.8, ls="--")
    ax.text(3.1, 0.51, "chance", ha="left", va="bottom", fontsize=6.5, color=INK_SECONDARY)
    for frames in (4, 16):
        ax.axvline(frames, color=AXIS, lw=0.8, ls=":")
        ax.text(frames, 0.03, f"{frames} frames", rotation=90, ha="right", va="bottom", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.set_xlim(3, horizons[-1] * 1.2)
    ax.set_xticks(horizons, horizons)
    ax.minorticks_off()
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Memory horizon L (corridor steps)")
    ax.set_ylabel("Success rate (IQM)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(E2_ORDER), columnspacing=1.0, handlelength=1.8)

    windows = sorted({key[1] for _, key in cells if key[0] == "LSTM"})
    grid = [[next((row["iqm"] for row in rows if (row["L"], row["series"], row["window"]) == (L, "LSTM", w)), float("nan"))
             for L in horizons] for w in windows]
    # From chance up: a memoryless policy already sits at 0.5, the printed values stay the real ones below it
    heatmap(ax_windows, grid, SEQUENTIAL, Normalize(0.5, 1.0), [f"BPTT {w}" for w in windows], horizons)
    ax_windows.set_xlabel("Memory horizon L")
    ax_windows.set_title("LSTM success rate (IQM)")

    e2_table(sweep_dir, storage / "figures" / str(cfg.timestamp), int(figure.get("table_L", 128)))
    return "fig3", fig, rows


# -- E3: information vs memory --

def figure_e3(sweep_dir: Path, storage: Path, figure, rng, cfg):
    n_boot = int(figure.get("n_boot", 2000))
    alpha = float(figure.get("alpha", 0.01))

    def key_of(fields):
        condition = E3_CONDITIONS.get((fields["ginfo"], fields["pinfo"], fields["ggrid"]))
        return None if condition is None else (condition, fields["backbone"])

    cells = collect(sweep_dir, "auc", key_of)
    rooms = ordered_rooms({room for room, _ in cells})
    conditions = list(E3_CONDITIONS.values())
    backbones = [b for b in BACKBONES if any(key[1] == b for _, key in cells)]

    rows = [{"room": room, "condition": key[0], "backbone": key[1], **score(values, rng, n_boot)}
            for (room, key), values in cells.items()]
    # Memory gain per condition, Holm over every comparison of the figure as in seed_stats.py
    gains = []
    for room in rooms:
        for condition in conditions:
            lstm, mlp = cells.get((room, (condition, "lstm"))), cells.get((room, (condition, "mlp")))
            if lstm is None or mlp is None:
                continue
            gain, low, high = iqm_difference(lstm, mlp, rng, n_boot)
            p_value = float(ttest_ind(lstm, mlp, equal_var=False).pvalue) if min(len(lstm), len(mlp)) > 1 else float("nan")
            gains.append({"room": room, "condition": condition, "backbone": "lstm-mlp", "n": min(len(lstm), len(mlp)),
                          "iqm": gain, "ci_low": low, "ci_high": high, "p_welch": p_value})
    # Two configs that both sit at exactly 0 or 1 give p = NaN: no evidence of a difference
    for row, adjusted in zip(gains, holm([1.0 if np.isnan(g["p_welch"]) else g["p_welch"] for g in gains])):
        row["p_holm"] = adjusted
        row["significant"] = bool(adjusted < alpha)

    bound = max([abs(g["iqm"]) for g in gains] + [0.05])
    # Shared by the panels, and from just below the worst cell so the conditions can be told apart
    floor = np.floor(min(row["iqm"] for row in rows) * 10) / 10
    fig, axes = plt.subplots(2, len(rooms), figsize=(DOUBLE_COLUMN, 1.1 + 0.3 * (len(backbones) + 1)), squeeze=False,
                             layout="constrained", sharex=True, height_ratios=[len(backbones), 1])
    for column, room in enumerate(rooms):
        grid = [[next((r["iqm"] for r in rows if (r["room"], r["condition"], r["backbone"]) == (room, c, b)), float("nan"))
                 for c in conditions] for b in backbones]
        heatmap(axes[0, column], grid, SEQUENTIAL, Normalize(floor, 1.0), [BACKBONES[b] for b in backbones], conditions)
        axes[0, column].set_title(ROOM_LABELS.get(room, room))

        room_gains = {g["condition"]: g for g in gains if g["room"] == room}
        gain_row = [[room_gains[c]["iqm"] if c in room_gains else float("nan") for c in conditions]]
        marks = [[c in room_gains and room_gains[c]["significant"] for c in conditions]]
        heatmap(axes[1, column], gain_row, DIVERGING, Normalize(-bound, bound), ["LSTM − MLP"], conditions, marks)
        axes[1, column].tick_params(labelbottom=True)
        if column > 0:
            axes[0, column].tick_params(labelleft=False)
            axes[1, column].tick_params(labelleft=False)
    return "fig4", fig, rows + gains


# -- E4: effect of every level, by tier --

def load_promotion(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No {path}: run scripts/e4_promotion.py on this sweep first")
    with open(path) as f:
        return json.load(f)

def rank_scores(promotion, room):
    """{level config: mean AUC in the room} over all the levels of a promotion analysis. Keyed on the config, not
    the label: labels are relative to each stage's base and repeat across blocks, so they don't match across stages."""
    return {level["config"]: level["mean_per_room"][room] for block in promotion["themes"].values()
            for level in block["levels"] if room in level["mean_per_room"]}

def rank_tau(scores_a, scores_b):
    """Kendall tau-b over the levels both rankings have, with their number."""
    shared = sorted(set(scores_a) & set(scores_b))
    if len(shared) < 3:
        return float("nan"), len(shared)
    return float(kendalltau([scores_a[s] for s in shared], [scores_b[s] for s in shared]).statistic), len(shared)

def figure_e4(sweep_dir: Path, storage: Path, figure, rng, cfg):
    n_boot = int(figure.get("n_boot", 2000))
    promotion = load_promotion(Path(figure.get("promotion") or storage / "promotion" / f"{cfg.timestamp}.json"))
    per_room = load_per_seed(sweep_dir, "auc")
    rooms = ordered_rooms(promotion["rooms"])

    # Forest plot: one row per level, grouped by the block that shares a reference. The "unmatched" blocks mix
    # the level's effect with how it differs from the baseline, the plan doesn't use their ranking
    rows, y_labels, headers, y = [], [], [], 0
    for block, content in promotion["themes"].items():
        if "unmatched" in block:
            continue
        reference = next(level for level in content["levels"] if level["label"] == content["reference"])
        headers.append(y)
        y_labels.append(block.replace(":", " "))
        y += 1
        for level in content["levels"]:
            if level is reference:
                continue
            for room in rooms:
                x, ref = (per_room.get(room, {}).get(c) for c in (level["config"], reference["config"]))
                if x is None or ref is None:
                    continue
                effect, low, high = iqm_difference(np.array(list(x.values())), np.array(list(ref.values())), rng, n_boot)
                rows.append({"block": block, "level": level["label"], "room": room, "y": y, "n": len(x), "n_reference": len(ref),
                             "effect": effect, "ci_low": low, "ci_high": high, "kept": level.get("kept") or "dropped"})
            y_labels.append(level["label"].replace(":", " "))
            y += 1

    # Kendall tau between neighbouring rooms, and between the stages on the same room when asked. A later stage
    # only holds survivors, so its tau has a restricted range (plan: probe panel)
    taus = []
    for room_a, room_b in zip(rooms, rooms[1:]):
        tau, n = rank_tau(rank_scores(promotion, room_a), rank_scores(promotion, room_b))
        taus.append({"comparison": f"{ROOM_LABELS.get(room_a, room_a)} / {ROOM_LABELS.get(room_b, room_b)}", "tau": tau, "n": n})
    if figure.get("compare"):
        other = load_promotion(Path(figure.get("compare")))
        for room in sorted(set(rooms) & set(other["rooms"])):
            tau, n = rank_tau(rank_scores(promotion, room), rank_scores(other, room))
            taus.append({"comparison": f"{ROOM_LABELS.get(room, room)}, between stages", "tau": tau, "n": n})
    for row in taus:
        print(f"[E4] Kendall tau {row['comparison']}: {row['tau']:.2f} over {row['n']} levels")
    out_dir = storage / "figures" / str(cfg.timestamp)
    save_rows(taus, out_dir / "fig2_right_tau.csv")

    fig, axes = plt.subplots(1, len(rooms), figsize=(SINGLE_COLUMN + 1.6, 1.0 + 0.135 * y), layout="constrained",
                             sharey=True, sharex=True, squeeze=False)
    colour = SERIES["MLP"][0]
    for ax, room in zip(axes[0], rooms):
        ax.axvline(0.0, color=INK_SECONDARY, lw=0.8)
        for row in (row for row in rows if row["room"] == room):
            ax.plot([row["ci_low"], row["ci_high"]], [row["y"], row["y"]], color=colour, lw=LINE_WIDTH, solid_capstyle="round")
            ax.plot(row["effect"], row["y"], "o", color=colour, ms=3.5, mec="white", mew=0.4)
        ax.set_title(ROOM_LABELS.get(room, room))
        ax.grid(axis="y", visible=False)
    fig.supxlabel("AUC effect against the reference (IQM difference)", fontsize=8, color=INK_SECONDARY)
    axes[0, 0].set_yticks(range(y), y_labels, fontsize=6)
    axes[0, 0].set_ylim(y - 0.5, -0.5)
    for tick, row in zip(axes[0, 0].get_yticklabels(), range(y)):
        if row in headers:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    if taus:
        fig.suptitle("Kendall τ: " + ", ".join(f"{row['comparison']} {row['tau']:.2f} (n={row['n']})" for row in taus),
                     fontsize=6.5, color=INK_SECONDARY)
    return "fig2_right", fig, rows


def save_rows(rows, path: Path):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

FIGURES = {"e1": figure_e1, "e1b": figure_e1b, "e2": figure_e2, "e3": figure_e3, "e4": figure_e4}


@hydra.main( config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    figure = cfg.get("figure", {})
    if figure.get("name") not in FIGURES:
        raise ValueError(f"Pick a figure: +figure.name={'|'.join(FIGURES)}")

    storage = Path(cfg.storage)
    sweep_dir = storage / "eval_metrics" / str(cfg.timestamp)
    out_dir = storage / "figures" / str(cfg.timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(figure.get("seed", 0)))

    logging.getLogger("fontTools").setLevel(logging.WARNING)   # Hydra's root logger would print every subset step
    paper_style()
    name, fig, rows = FIGURES[figure.name](sweep_dir, storage, figure, rng, cfg)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    save_rows(rows, out_dir / f"{name}.csv")
    print(f"Saved {out_dir / name}.pdf, .png and .csv (bootstrap seed {int(figure.get('seed', 0))})")

    if cfg.wandb.mode != "disabled":
        with wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, group=cfg.wandb.group,
                        name=f"{name}_{cfg.timestamp}", job_type="eval",
                        config={"storage": str(cfg.storage), "timestamp": str(cfg.timestamp), "figure": figure.name},
                        tags=["eval", "paper_figures", f"timestamp:{cfg.timestamp}"], mode=cfg.wandb.mode):
            wandb.log({f"figures/{name}": wandb.Image(str(out_dir / f"{name}.png"), caption=figure.name)})


if __name__ == "__main__":
    main()
