"""Training curves from W&B (Eval Success Rate per eval, mean ± SE over seeds) for the supervisor report."""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
C = {"MLP": "#2a78d6", "LSTM": "#eb6834", "GRU": "#1baf7a", "FS4": "#eda100", "FS16": "#e87ba4",
     "LSTM-BPTT256": "#9e3a12", "grey": "#8a8984"}
BLUES = ["#0d366b", "#184f95", "#2a78d6", "#6da7ec"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8})
runs = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "wb_runs.pkl", "rb"))
OUT = "."


def cfg(r, k, d=None):
    return r["config"].get(k, d)


def padded(r):
    """Success at every eval of the full budget, last value carried forward (as the padded AUC does)."""
    n, every = int(cfg(r, "algorithm.n_iterations", 1000)), int(cfg(r, "algorithm.eval_interval", 5))
    grid = np.arange(every, n + 1, every)
    if not r["curve"]:
        return grid, None
    it, s = np.array(r["curve"], dtype=float).T
    idx = np.searchsorted(it, grid, side="right") - 1
    y = np.where(idx >= 0, s[np.clip(idx, 0, None)], 0.0)
    return grid, y


def band(ax, rs, colour, label, ls="-", smooth=5):
    curves = [padded(r) for r in rs]
    curves = [(g, y) for g, y in curves if y is not None]
    if not curves:
        return
    g = curves[0][0]
    Y = np.stack([y for _, y in curves])
    if smooth > 1:   # moving average over evals, eval success is noisy (10 episodes per eval)
        k = np.ones(smooth) / smooth
        Y = np.stack([np.convolve(np.pad(y, (smooth - 1, 0), mode="edge"), k, mode="valid") for y in Y])
    m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else 0 * Y[0]
    ax.plot(g, m, color=colour, ls=ls, label=f"{label} (n={len(Y)})")
    ax.fill_between(g, m - se, m + se, color=colour, alpha=0.15, lw=0)


def finish(ax, title, xlabel=True):
    ax.set_title(title, loc="left")
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3))) if "twin" in title or "L =" in title else None
    if xlabel:
        ax.set_xlabel("PPO iteration")


def room(r):
    return cfg(r, "env.room_path", "").split("/")[-1].replace(".json", "")


def backbone(r):
    b = cfg(r, "backbone.name")
    if b == "mlp":
        return f"FS{cfg(r, 'observation.frame_stack.size')}" if cfg(r, "observation.frame_stack.enabled") else "MLP"
    if b == "gru":
        return "GRU"
    w = cfg(r, "backbone.bptt_window_size")
    return "LSTM" if w == 16 else f"LSTM-BPTT{w}"


def pick(**want):
    out = []
    for r in runs:
        if all((f(r) if callable(f) else None) == v if callable(f) else cfg(r, k) == v
               for k, (f, v) in [(k, (v if isinstance(v, tuple) else (None, v))) for k, v in want.items()]):
            out.append(r)
    return out


def sel(rs, **conds):
    return [r for r in rs if all(fn(r) == v for fn, v in conds.values())]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=220)
    plt.close(fig)
    print("saved", name)


# ---------- E1: place-code audit, C_maze and medium_maze ----------
e1 = [r for r in runs if r["group"] == "e1_place_code_audit"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, rm in zip(axes, ["C_maze", "medium_maze"]):
    for b in ["MLP", "LSTM"]:
        for (col, goal), ls, lab in [(("global", "fixed"), (0, (2, 2)), "colour on, fixed goal"),
                                     (("off", "random"), "-", "colour off, random goal")]:
            rs = [r for r in e1 if room(r) == rm and backbone(r) == b and cfg(r, "env.colour_mode") == col
                  and cfg(r, "env.goal_mode") == goal]
            band(ax, rs, C[b], f"{b}, {lab}", ls=ls)
    finish(ax, rm)
axes[0].set_ylabel("Eval success rate")
axes[1].legend(loc="lower right", fontsize=7)
save(fig, "wb_e1")

# ---------- E2: twin rooms, success vs iteration per L ----------
e2 = [r for r in runs if r["group"] == "e2_twin_rooms" and cfg(r, "timestamp") == 20261110145407]
order = ["MLP", "FS16", "LSTM", "LSTM-BPTT256", "GRU"]
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), sharey=True)
for ax, L in zip(axes, [32, 64, 128, 256]):
    for b in order:
        band(ax, [r for r in e2 if room(r) == f"twin_rooms_L{L}" and backbone(r) == b], C[b],
             {"LSTM": "LSTM-BPTT16", "GRU": "GRU (BPTT64)"}.get(b, b))
    finish(ax, f"twin rooms, L = {L}")
axes[0].set_ylabel("Eval success rate")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, [x.split(" (n=")[0] for x in l], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
save(fig, "wb_e2")

# ---------- Layer check (L = 64, LSTM-BPTT16) ----------
layers = [r for r in runs if r["group"] == "e2_twin_rooms" and cfg(r, "timestamp") == 20261112142837]
ref = [r for r in e2 if room(r) == "twin_rooms_L64" and backbone(r) == "LSTM"]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for n, col in [(1, "#f3a37f"), (2, C["LSTM"]), (3, "#9e3a12")]:
    rs = ref if n == 2 else [r for r in layers if cfg(r, "backbone.lstm_num_layers") == n]
    band(ax, rs, col, f"{n} layer{'s' if n > 1 else ''}")
finish(ax, "LSTM-BPTT16 depth, L = 64")
ax.set_ylabel("Eval success rate")
ax.legend(loc="lower right", fontsize=7)
save(fig, "wb_layers")

# ---------- E3T: told or remembered ----------
e3t = [r for r in runs if r["group"] == "e3_twin_rooms"]


def condition(r):
    ci, n = cfg(r, "env.information.cue_info"), cfg(r, "env.information.cue_interval")
    if ci == "every":
        return f"C{n}"
    if cfg(r, "env.information.goal_info", "none") != "none":
        return "G1"
    if cfg(r, "env.information.position_info", "none") != "none":
        return "P1"
    return "I0"


fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7), sharey=True)
for ax, b in zip(axes, ["MLP", "FS16", "LSTM"]):
    for cond, col, ls in [("C1", BLUES[0], "-"), ("C4", BLUES[1], "-"), ("C16", BLUES[2], "-"), ("C64", BLUES[3], "-"),
                          ("I0", C["grey"], "-"), ("G1", C["GRU"], (0, (2, 2))), ("P1", C["FS4"], (0, (2, 2)))]:
        band(ax, [r for r in e3t if backbone(r) == b and condition(r) == cond], col, cond, ls=ls)
    finish(ax, f"twin rooms L = 32, {'LSTM-BPTT16' if b == 'LSTM' else b}")
axes[0].set_ylabel("Eval success rate")
h, l = axes[0].get_legend_handles_labels()
names = {"C1": "cue every step", "C4": "every 4", "C16": "every 16", "C64": "every 64", "I0": "never",
         "G1": "told the goal", "P1": "told xy"}
fig.legend(h, [names[x.split(" (n=")[0]] for x in l], loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
save(fig, "wb_e3t")
