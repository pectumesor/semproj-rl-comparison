"""
Collect the material behind the supervisor report ("Perceive, Remember, or Be Told - direction and results") in one
W&B group: one run per report section, each with

    Figures/   the report's figures (extracted from the PDF), with their captions
    Rooms/     a picture of every room the section trained in (walls, spawn regions, goal regions, cue walls)
    Curves/    per room, the eval success curve of every config (mean over seeds) and of every seed
    Tables/    per seed (AUC, eval success, diagnostics success where E2 ran) and per config (IQM, solved seeds)
    Rollouts/  videos of whole evaluation episodes from the saved best.pt checkpoints: in the twin rooms one
               episode per cue, elsewhere three episodes, with the trail, the rays and the outcome drawn in

The training runs stay in their own sweep groups; this only reads the seed folders the sweeps saved under
<root>/<sweep>/{models,eval_metrics}/<timestamp>/<room>/<storage name>/seed_<s>/. Checkpoints are rebuilt from the
Hydra overrides the sweep saved for each seed, composed with today's configs (whose defaults reproduce the old
behaviour), and only used if the composed storage name points at the saved seed folder.

    python scripts/report_to_wandb.py --section E2 --root /cluster/scratch/eazevedo/semproj \
        --figures <dir with the PDF's figures> --group report_nov19

Sections: overview, E1, E1b, E2, E2K4, E3, E3T, E4 (see SECTIONS). --videos none skips the rollouts.
"""
import sys
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import re
import yaml
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import trim_mean
from omegaconf import open_dict
from hydra import initialize_config_dir, compose
import wandb

import utils  # noqa: F401  -- must come before envs: envs -> utils.geometry -> utils -> algorithms -> envs is circular
from utils import create_ppo_agent, create_environment, seed_metric_summary
from utils.geometry import goal_regions_json
from utils.rollout_video import FPS, record, room_picture
from envs import compute_num_rays, compute_proprio_dim
from scripts.train import run_names

# Sweeps: (storage folder under --root, timestamp, label used in the panel names, rollout mode).
# Rollouts "extremes" = best and worst seed of every config (twin rooms, where solved vs unsolved is the result),
# "median" = the median seed by AUC, None = curves and tables only
SECTIONS = {
    "overview": {"title": "Overview and testbed", "sweeps": [],
                 "figures": [("fig-003-000.png", "Testbed: the twin rooms (L = 32) and the three mazes used in E1-E4")]},
    "E1": {"title": "E1 place-code audit", "sweeps": [("e1_place_code_audit", "20261108114624", "E1", "median")],
           "figures": [("fig-004-002.png", "E1 learning curves: colour on/off, fixed/random goal, C_maze and medium_maze"),
                       ("fig-005-004.png", "E1 summary (paper Fig. 2 left): AUC IQM per room and condition, memory gain vs aliasing")]},
    "E1b": {"title": "E1b field of view", "sweeps": [("e1b_fov", "20261108114624", "E1b", "median")],
            "figures": [("fig-005-006.png", "E1b: success AUC (IQM) vs field of view in C_maze")]},
    "E2": {"title": "E2 twin rooms", "sweeps": [("e2_twin_rooms", "20261110145407", "E2", "extremes"),
                                                ("e2_twin_rooms", "20261112142837", "E2 layers", "extremes")],
           "figures": [("fig-006-008.png", "E2 summary (paper Fig. 3): success vs corridor length per backbone, LSTM window heatmap"),
                       ("fig-006-010.png", "E2 learning curves per corridor length L"),
                       ("fig-006-012.png", "E2 layer check: LSTM-BPTT16 with 1, 2 and 3 layers at L = 64")]},
    "E2K4": {"title": "E2 four rooms (K = 4)", "sweeps": [("e2_twin_rooms_k4", "20261118095532", "K4", "extremes")],
             "figures": [("page-15.png", "Update Nov 19: E2 with four rooms (report page 15)")]},
    "E3": {"title": "E3 information sources", "sweeps": [("e3_information", "20261110205630", "E3", "median")],
           "figures": [("fig-007-014.png", "E3 summary (paper Fig. 4): conditions x backbones, four_room and C_maze")]},
    "E3T": {"title": "E3T told or remembered", "sweeps": [("e3_twin_rooms", "20261112133519", "E3T", "extremes"),
                                                         ("e3_twin_rooms", "20261115173643", "E3T random phase", "extremes"),
                                                         ("e3_twin_rooms", "20261117110027", "E3T L64", "extremes")],
            "figures": [("fig-008-016.png", "E3T summary: success (IQM) vs telling interval per backbone, told goal / xy"),
                        ("fig-008-018.png", "E3T learning curves per telling interval")]},
    "E4": {"title": "E4 staged ablation", "sweeps": [("e4_stage1", "20261102095921", "Stage 1", None),
                                                     ("e4_stage2", "20261104133601", "Stage 2", None),
                                                     ("e4_stage3", "20261116094617", "Stage 3", "median")],
           "figures": [("fig-012-020.png", "E4 Stage 2 learning curves of the theme winners, medium_maze and hard_maze")]},
}

# Keys of the storage-name fields (train.run_names / variant_name), to split a folder name into its settings
NAME_KEYS = ["encoder", "backbone", "frame_stack", "grid_cell", "gc", "auxiliary_head", "ah", "bptt", "colour", "goal",
             "ginfo", "pinfo", "ggrid", "fov", "act", "cpen", "gamma", "ep", "mask", "width", "lr", "aux", "depth",
             "layers", "jshape", "cue", "ph", "modact", "mb", "bs"]
SPLIT = re.compile(r"_(?=(?:" + "|".join(NAME_KEYS) + r"):)")
SOLVED = 0.9                    # diagnostics success that counts a twin-room seed as solved (plan, E2)
# Categorical line colours in this fixed order (dataviz reference palette, light surface): every adjacent pair passes
# the colour-blind (ΔE >= 9.1) and normal-vision (ΔE >= 19.6) separation checks of the dataviz validator. A panel
# never needs a 9th colour: more configs split into small multiples (curve_layout)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"


def parse_name(name: str) -> dict:
    """Storage name -> {key: value}, with the short keys of over-long names (gc, ah) under their long names."""
    fields = dict(token.split(":", 1) for token in SPLIT.split(name) if ":" in token)
    for short, long in (("gc", "grid_cell"), ("ah", "auxiliary_head")):
        if short in fields:
            fields[long] = fields.pop(short)
    return fields

def arch_label(fields: dict) -> str:
    """Short architecture name as in the report: MLP, FS16, LSTM-BPTT64, GRU-BPTT64, CNN+..."""
    backbone = fields.get("backbone", "?")
    stack = fields.get("frame_stack", "False")
    if backbone == "mlp":
        label = f"FS{stack[4:]}" if stack.startswith("True") else "MLP"
    else:
        label = f"{backbone.upper()}-BPTT{fields.get('bptt', '?')}"
    if fields.get("encoder", "mlp") != "mlp":
        label = f"{fields['encoder'].upper()}+{label}"
    if fields.get("auxiliary_head", "False") == "True":
        label += "+aux"
    return label

def setting(key: str, value: str) -> str:
    """One setting as the report names it."""
    if key == "cue":
        return "never told" if value == "0" else f"told every {value}"
    if key == "ph":
        return "random phase" if value == "rnd" else f"phase {value}"
    # Information slots: "none" is the default, so only a slot that is on is named
    if key in ("ginfo", "pinfo", "ggrid") and value in ("none", "False"):
        return ""
    if key == "ginfo" and value == "oracle":
        return "goal told"
    if key == "pinfo" and value == "xy":
        return "xy told"
    return f"{key}:{value}"

def config_parts(names: list) -> dict:
    """
    Storage name -> (architecture, other settings): the settings that differ between the names, without those that
    follow the architecture (E1-E3 give the MLP and the LSTM their own lr, E2 its window/minibatch pairs) unless they
    are needed to tell two configs apart. A setting missing from a name (older folders) is its default and not shown.
    """
    fields = {n: parse_name(n) for n in names}
    archs = {n: arch_label(f) for n, f in fields.items()}
    arch_keys = {"encoder", "backbone", "frame_stack", "bptt", "auxiliary_head"}
    varying = [k for k in NAME_KEYS if k not in arch_keys and len({f.get(k, "-") for f in fields.values()}) > 1]
    follows_arch = [k for k in varying if len(set(archs.values())) > 1
                    and all(len({fields[n].get(k, "-") for n in names if archs[n] == a}) == 1 for a in set(archs.values()))]
    def rest(f, keys):
        words = [w for w in (setting(k, f[k]) for k in keys if k in f) if w]
        # A run told the goal or position isn't told the cue (E3T), "never told" would only repeat that
        return " ".join(w for w in words if not (w == "never told" and ("goal told" in words or "xy told" in words)))
    make = lambda keys: {n: (archs[n], rest(f, keys)) for n, f in fields.items()}
    parts = make([k for k in varying if k not in follows_arch])
    # The config without any of the varied settings (E3's no-information condition) still needs a name
    if any(r for _, r in parts.values()):
        parts = {n: (a, r or "baseline") for n, (a, r) in parts.items()}
    if len(set(parts.values())) < len(parts):
        parts = make(varying)
    # Two names can still map to one label if they differ only in an architecture field the label drops
    if len(set(parts.values())) < len(parts):
        parts = {n: (a, f"{r} #{i}".strip()) for i, (n, (a, r)) in enumerate(parts.items())}
    return parts

def config_labels(names: list) -> dict:
    """Storage name -> short label, the architecture and the settings that tell the configs apart."""
    return {n: f"{a} {r}".strip() for n, (a, r) in config_parts(names).items()}


# ---------- seed folders ----------

def load_seeds(root: Path, sweep: str, timestamp: str):
    """{room: {storage name: {seed: {"curve", "summary", "diag"}}}} of one sweep."""
    rooms = {}
    for seed_dir in sorted((root / sweep / "eval_metrics" / timestamp).glob("*/*/seed_*")):
        curve_path = seed_dir / "success_curve.npy"
        if not curve_path.is_file():
            continue
        diag_path = seed_dir / "e2_diagnostics.json"
        diag = json.load(open(diag_path)) if diag_path.is_file() else None
        room, name, seed = seed_dir.parent.parent.name, seed_dir.parent.name, int(seed_dir.name.split("_")[1])
        rooms.setdefault(room, {}).setdefault(name, {})[seed] = {
            "curve": np.load(curve_path), "summary": seed_metric_summary(seed_dir),
            "diag": diag["memory"]["success"] if diag else None}
    return rooms

def seed_score(entry: dict) -> float:
    """Diagnostics success where it ran (1000 deterministic episodes), otherwise the budget-padded AUC."""
    return entry["diag"] if entry["diag"] is not None else entry["summary"].get("Metrics/AUC", 0.0)

def padded(curves: list) -> np.ndarray:
    """(S, N) curves padded with their last value to the longest one, as the AUC pads early-stopped runs."""
    n = max(len(c) for c in curves)
    return np.stack([np.concatenate([c, np.full(n - len(c), c[-1])]) for c in curves])


# ---------- checkpoints ----------

def index_overrides(root: Path, sweep: str, timestamp: str) -> dict:
    """(room, storage name, seed) -> Hydra overrides of every training run of the sweep that saved them."""
    files = list((root / sweep).glob("**/.hydra/overrides.yaml"))
    home = Path.home() / "semproj-rl-comparison" / "multirun"
    if home.is_dir():
        files += [f for f in home.glob("**/.hydra/overrides.yaml") if f"timestamp={timestamp}" in f.read_text()]
    index = {}
    for path in files:
        overrides = yaml.safe_load(path.read_text())
        # Training runs only: the diagnostics' multiruns save the same overrides without a seed
        if f"timestamp={timestamp}" not in overrides or not any(o.startswith("seed=") for o in overrides) \
                or any(o.startswith("+diagnostics") for o in overrides):
            continue
        cfg = compose(config_name="base", overrides=overrides)
        _, storage_name = run_names(cfg)
        index[(Path(cfg.env.room_path).stem, storage_name, int(cfg.seed))] = overrides
    return index

def load_agent(overrides: list, checkpoint: Path):
    cfg = compose(config_name="base", overrides=overrides + ["wandb.mode=disabled"])
    with open_dict(cfg):
        cfg.env.proprio_dim = compute_proprio_dim(cfg)
    num_rays = compute_num_rays(cfg.env.fov, cfg.env.ray_density)
    ray_dim = np.array([cfg.env.ray_encoding, num_rays])
    agent = create_ppo_agent(ray_dim=ray_dim, cfg=cfg)
    agent.load_model(checkpoint, "cpu", optim.Adam(agent.parameters()))
    agent.eval()
    env = create_environment(cfg=cfg, agent=agent, num_rays=num_rays, ray_dim=ray_dim, num_envs=1, device="cpu")
    return cfg, agent, env


# ---------- curves ----------

def natural(text: str):
    """Sort key with numbers compared as numbers; "never told", then goal / xy told after the intervals (E3T figure)."""
    text = text.replace("never told", "told every 999999").replace("goal told", "~goal").replace("xy told", "~xy")
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]

def arch_order(arch: str):
    """Memoryless first, then frame stacks, LSTM by window, GRU; CNN encoders and aux heads after their base."""
    base = arch.removeprefix("CNN+").removesuffix("+aux")
    kind = next((i for i, k in enumerate(["MLP", "FS", "LSTM", "GRU"]) if base.startswith(k)), 4)
    return (arch.startswith("CNN+"), kind, natural(base), arch.endswith("+aux"))

def family(arch: str) -> str:
    return arch.removesuffix("+aux")

def wrap(text: str, width: int = 48) -> str:
    """Line breaks (<br>) between words, so long subplot titles don't run into the next subplot's."""
    lines = [""]
    for word in text.split():
        if lines[-1] and len(lines[-1]) + len(word) + 1 > width:
            lines.append("")
        lines[-1] = f"{lines[-1]} {word}".strip()
    return "<br>".join(lines)

def curve_layout(parts: list):
    """
    How every room's panel of one sweep is drawn, from the (architecture, settings) of all its configs, so a config
    keeps its colour in every room. Returns (mode, colour keys in palette order):
        arch    a subplot per architecture, a colour per setting (both vary, at most 8 settings: E1, E1b, E3T)
        single  one panel, a colour per config                   (at most 8 configs, e.g. E2's backbones)
        rest    a subplot per setting, a colour per architecture (at most 8 architectures, e.g. E3's 9 conditions)
        config  a subplot per config, a colour per architecture family, the architecture without its aux head
                (E4 Stage 2: 9 architectures x 22 settings, 6 families)
    """
    label = lambda a, r: f"{a} {r}".strip()
    archs = sorted({a for a, _ in parts}, key=arch_order)
    rests = sorted({r for _, r in parts}, key=natural)
    labels = sorted({label(a, r) for a, r in parts}, key=lambda l: next((arch_order(a), natural(r)) for a, r in parts
                                                                       if label(a, r) == l))
    # Settings within an architecture share its subplot, so related lines aren't scattered over unrelated hues
    if len(archs) > 1 and 1 < len(rests) <= len(PALETTE):
        return "arch", rests
    if len(labels) <= len(PALETTE):
        return "single", labels
    if len(archs) <= len(PALETTE):
        return "rest", archs
    families = sorted({family(a) for a in archs}, key=arch_order)
    return "config", families if len(families) <= len(PALETTE) else []

def curve_figure(title: str, entries: list, mode: str, keys: list, chance: float = None):
    """
    Eval success over training: a thick line per config (mean over seeds), thin lines of the same colour for its
    seeds. entries: dicts with arch, rest, label, seeds {seed: (N,) curve}, mean (N,), x (N,).
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    colour_key = {"single": lambda e: e["label"], "arch": lambda e: e["rest"], "rest": lambda e: e["arch"],
                  "config": lambda e: family(e["arch"])}[mode]
    facet_key = {"single": lambda e: "", "arch": lambda e: e["arch"], "rest": lambda e: e["rest"],
                 "config": lambda e: e["label"]}[mode]
    colour = {k: PALETTE[i] for i, k in enumerate(keys)}
    if not keys:   # more families than colours: one colour, the subplot titles name the configs
        colour_key = lambda e: ""
        colour = {"": PALETTE[0]}
    facet_order = {"arch": arch_order, "config": lambda f: next(arch_order(e["arch"]) + (natural(e["rest"]),)
                                                                 for e in entries if e["label"] == f)}
    facets = sorted({facet_key(e) for e in entries}, key=facet_order.get(mode, natural))
    cols = min(len(facets), 3)
    rows = int(np.ceil(len(facets) / cols))
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes="all", shared_yaxes=True,
                        subplot_titles=[wrap(f) for f in facets] if len(facets) > 1 else None,
                        horizontal_spacing=0.05, vertical_spacing=(0.35 / rows if rows > 3 else 0.1) if rows > 1 else 0.0)
    shown = set()
    for e in sorted(entries, key=lambda e: keys.index(colour_key(e)) if colour_key(e) in keys else 0):
        index = facets.index(facet_key(e))
        row, col = index // cols + 1, index % cols + 1
        key, c = colour_key(e), colour[colour_key(e)]
        for seed, curve in e["seeds"].items():
            # Plain lists: plotly 6 encodes numpy arrays as base64 typed arrays, which older plotly.js can't read
            fig.add_trace(go.Scatter(x=list(map(int, e["x"])), y=list(map(float, curve)), mode="lines",
                                     line=dict(color=c, width=1), opacity=0.22,
                                     legendgroup=key, showlegend=False,
                                     hovertemplate=f"{e['label']}, seed {seed}<br>iteration %{{x}}<br>success %{{y:.2f}}<extra></extra>"),
                          row=row, col=col)
        fig.add_trace(go.Scatter(x=list(map(int, e["x"])), y=list(map(float, e["mean"])), mode="lines",
                                 line=dict(color=c, width=2.5), name=key,
                                 legendgroup=key, showlegend=bool(key) and key not in shown,
                                 hovertemplate=f"{e['label']}, mean of {len(e['seeds'])} seeds<br>iteration %{{x}}<br>success %{{y:.2f}}<extra></extra>"),
                      row=row, col=col)
        shown.add(key)
    for index in range(len(facets)):
        if chance is not None:
            # Label only the first: an empty annotation_text shows plotly's "new text" placeholder
            label_it = dict(annotation_text="chance", annotation_font_color=MUTED) if index == 0 else {}
            fig.add_hline(y=chance, line=dict(color=MUTED, width=1, dash="dot"), row=index // cols + 1,
                          col=index % cols + 1, **label_it)
    height = 420 if rows == 1 else 230 * rows + 150
    fig.update_yaxes(range=[-0.02, 1.02], gridcolor=GRID, zeroline=False, tickformat=".1f")
    fig.update_yaxes(title_text="eval success", col=1)
    # Every subplot over the whole budget, also where all its seeds stopped early
    fig.update_xaxes(gridcolor=GRID, zeroline=False, range=[0, max(int(e["x"][-1]) for e in entries)])
    fig.update_xaxes(title_text="PPO iteration", row=rows)
    fig.update_annotations(font_size=12 if mode != "config" else 10, font_color=INK)
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left", y=0.99, yanchor="top"),
                      template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="Inter, DejaVu Sans, Arial, sans-serif", size=12, color=MUTED),
                      # Legend between the title and the plot, clear of the axis titles
                      legend=dict(orientation="h", x=0, y=1.0 + 60 / height, yanchor="bottom", font=dict(color=INK)),
                      height=height, margin=dict(l=60, r=20, t=120, b=50),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    return fig


# ---------- W&B ----------

def log_sweep(run, root: Path, sweep: str, timestamp: str, label: str, videos: str, video_dir: Path, args):
    rooms = load_seeds(root, sweep, timestamp)
    print(f"[report] {label}: {sum(len(s) for r in rooms.values() for s in r.values())} seeds in {len(rooms)} rooms")
    index = index_overrides(root, sweep, timestamp) if videos and args.videos != "none" else {}
    seed_rows, config_rows = [], []
    # One colour per config across every room of the sweep. Rooms with the same configs share a layout; the sweeps
    # here have the same configs in every room
    all_parts = [p for configs in rooms.values() for p in config_parts(list(configs)).values()]
    layouts = {room: curve_layout(all_parts) for room in rooms}
    for room, configs in sorted(rooms.items()):
        labels = config_labels(list(configs))
        room_path = f"rooms/{room}.json"
        if (ROOT_DIR / room_path).is_file():
            run.log({f"Rooms/{room}": wandb.Image(room_picture(str(ROOT_DIR / room_path)), caption=room)})

        # Curves: mean over seeds per config (thick) and every seed (thin, same colour), x = PPO iteration of the eval
        parts = config_parts(list(configs))
        entries = []
        for name, seeds in configs.items():
            curves = padded([s["curve"] for s in seeds.values()])
            entries.append({"arch": parts[name][0], "rest": parts[name][1], "label": labels[name], "mean": curves.mean(0),
                            "seeds": dict(zip(seeds, curves)), "x": np.arange(1, curves.shape[1] + 1) * 5})
        regions = goal_regions_json(str(ROOT_DIR / room_path)) if (ROOT_DIR / room_path).is_file() else None
        chance = 1.0 / len(regions[0]) if regions else None
        run.log({f"Curves/{label} {room}": wandb.Plotly(curve_figure(
            f"{label}, {room}: eval success (thick: mean over seeds, thin: seeds)", entries, *layouts[room], chance))})

        for name, seeds in sorted(configs.items(), key=lambda kv: labels[kv[0]]):
            aucs = np.array([s["summary"].get("Metrics/AUC", np.nan) for s in seeds.values()])
            finals = np.array([s["summary"].get("Metrics/Final Eval Success Rate", np.nan) for s in seeds.values()])
            diags = [s["diag"] for s in seeds.values() if s["diag"] is not None]
            for seed, s in sorted(seeds.items()):
                seed_rows.append([label, room, labels[name], seed, s["summary"].get("Metrics/AUC"),
                                  s["summary"].get("Metrics/Final Eval Success Rate"), s["diag"], name])
            config_rows.append([label, room, labels[name], len(seeds), float(trim_mean(aucs, 0.25)), float(np.nanmean(aucs)),
                                float(np.nanmean(finals)), float(np.mean(diags)) if diags else None,
                                f"{sum(d >= SOLVED for d in diags)}/{len(diags)}" if diags else None, name])

            if not index:
                continue
            ranked = sorted(seeds, key=lambda seed: seed_score(seeds[seed]), reverse=True)
            if videos == "median":
                picks = [(ranked[len(ranked) // 2], "median seed")]
            else:
                picks = [(ranked[0], "best seed")]
                if seed_score(seeds[ranked[0]]) - seed_score(seeds[ranked[-1]]) > 0.05:
                    picks.append((ranked[-1], "worst seed"))
            for seed, tag in picks:
                if args.max_videos is not None and len(list(video_dir.glob("*.mp4"))) >= args.max_videos:
                    break
                checkpoint = root / sweep / "models" / timestamp / room / name / f"seed_{seed}" / "best.pt"
                overrides = index.get((room, name, seed))
                if overrides is None or not checkpoint.is_file():
                    print(f"[report] no {'overrides' if overrides is None else 'checkpoint'} for {room}/{labels[name]}/seed_{seed}, skipping its video")
                    continue
                cfg, agent, env = load_agent(overrides, checkpoint)
                score = seed_score(seeds[seed])
                kind = "diagnostics success" if seeds[seed]["diag"] is not None else "AUC"
                title = f"{label} - {room} - {labels[name]}"[:70]
                frames, outcomes = record(agent, env, str(ROOT_DIR / cfg.env.room_path), cfg.backbone.name != "mlp", title)
                path = video_dir / f"{label}_{room}_{len(seed_rows)}_{seed}.mp4".replace(" ", "_")
                import imageio
                imageio.mimsave(str(path), frames, fps=FPS, macro_block_size=16)
                caption = f"{labels[name]}, seed {seed} ({tag}, {kind} {score:.2f}). " + "; ".join(outcomes)
                key = f"Rollouts/{label} {room}/{labels[name]} s{seed}"[:150]
                run.log({key: wandb.Video(str(path), format="mp4", caption=caption)})
                print(f"[report] video {key}: {len(frames)} frames, {caption}")
    return seed_rows, config_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True, choices=list(SECTIONS))
    parser.add_argument("--root", default="/cluster/scratch/eazevedo/semproj")
    parser.add_argument("--figures", required=True, help="folder with the figures extracted from the report PDF")
    parser.add_argument("--group", default="report_nov19")
    parser.add_argument("--videos", default="all", choices=["all", "none"])
    parser.add_argument("--project", default="semproj")
    parser.add_argument("--entity", default="eazevedo-eth-z-rich")
    parser.add_argument("--mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--max-videos", type=int, default=None, help="stop recording after this many (for tests)")
    args = parser.parse_args()

    section = SECTIONS[args.section]
    root = Path(args.root)
    video_dir = root / "report_wandb" / "videos" / args.section
    video_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(ROOT_DIR / "configs"), version_base=None), \
         wandb.init(entity=args.entity, project=args.project, group=args.group, job_type="report",
                    name=section["title"], mode=args.mode,
                    config={"section": args.section, "sweeps": [list(s[:3]) for s in section["sweeps"]]}) as run:

        for file, caption in section["figures"]:
            path = Path(args.figures) / file
            if path.is_file():
                run.log({f"Figures/{caption.split(':')[0]}": wandb.Image(str(path), caption=caption)})
            else:
                print(f"[report] figure {path} missing")

        if args.section == "overview":
            for room_path in sorted((ROOT_DIR / "rooms").glob("*.json")):
                if room_path.stem in ("two_room", "four_room", "C_maze", "medium_maze", "hard_maze",
                                      "twin_rooms_L32", "twin_rooms_K4_L32"):
                    run.log({f"Rooms/{room_path.stem}": wandb.Image(room_picture(str(room_path)), caption=room_path.stem)})

        seed_rows, config_rows = [], []
        for sweep, timestamp, label, videos in section["sweeps"]:
            s, c = log_sweep(run, root, sweep, timestamp, label, videos, video_dir, args)
            seed_rows += s
            config_rows += c
        if config_rows:
            run.log({"Tables/configs": wandb.Table(columns=["sweep", "room", "config", "seeds", "AUC IQM", "AUC mean",
                                                            "final eval success", "diagnostics success",
                                                            f"solved (>= {SOLVED})", "storage name"], data=config_rows),
                     "Tables/seeds": wandb.Table(columns=["sweep", "room", "config", "seed", "AUC", "final eval success",
                                                          "diagnostics success", "storage name"], data=seed_rows)})
        print(f"[report] {args.section} done: {run.url}")


if __name__ == "__main__":
    main()
