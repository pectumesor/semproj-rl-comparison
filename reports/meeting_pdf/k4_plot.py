"""E2 K = 4 training curves (eval success per eval, mean ± SE over seeds 100–104), same style as wb_plots.py."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
C = {"GRU-BPTT64": "#1baf7a", "LSTM-BPTT256": "#9e3a12", "LSTM-BPTT64": "#6250d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8})
runs = json.load(open(sys.argv[1]))
N_EVALS, EVERY, SMOOTH = 200, 5, 5


def backbone(arch):
    b = arch.split("backbone:")[1].split("_")[0]
    w = arch.split("bptt:")[1].split("_")[0]
    return f"{b.upper()}-BPTT{w}"


def band(ax, curves, colour, label):
    # Padded to the full budget with the last value (as the padded AUC), then a moving average over 5 evals
    Y = np.stack([np.pad(c, (0, N_EVALS - len(c)), mode="edge") for c in curves])
    k = np.ones(SMOOTH) / SMOOTH
    Y = np.stack([np.convolve(np.pad(y, (SMOOTH - 1, 0), mode="edge"), k, mode="valid") for y in Y])
    g = EVERY * np.arange(1, N_EVALS + 1)
    m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
    ax.plot(g, m, color=colour, label=label)
    ax.fill_between(g, m - se, m + se, color=colour, alpha=0.15, lw=0)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, L in zip(axes, [32, 128]):
    for b, colour in C.items():
        band(ax, [np.array(r["curve"]) for r in runs if r["room"] == f"twin_rooms_K4_L{L}" and backbone(r["arch"]) == b],
             colour, b)
    ax.set_title(f"four rooms (K = 4), L = {L}", loc="left")
    ax.set_ylim(0, 1.02)
    ax.axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("PPO iteration")
axes[0].set_ylabel("Eval success rate")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.tight_layout()
fig.savefig("wb_k4.png", dpi=220, bbox_inches="tight")
print("saved wb_k4")
