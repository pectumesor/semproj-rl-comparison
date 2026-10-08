import numpy as np
import torch
from .geometry import path_length
import wandb
from utils import generate_reference_trajectory
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, t
 
"""
Script with metric function to test the performance of trained policies

"""

def extract_trajectory_recurrent(agent, env, nr_runs):

    num_layers = agent.backbone_model.lstm.num_layers
    hidden_size = agent.backbone_model.lstm.hidden_size

    obs, info = env.reset()
    lstm_state = (
                    torch.zeros((num_layers, env.num_envs, hidden_size),
                                dtype=torch.float, device=env.device),
                    torch.zeros((num_layers, env.num_envs, hidden_size),
                                 dtype=torch.float, device=env.device)
                    )
    # Episode-start flag fed to the LSTM on the *next* step: True for an env that
    # was just reset (terminated *or* truncated), so SimpleLSTM.forward zeroes its
    # hidden state instead of leaking it into the fresh episode.
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    max_len = env.unwrapped.max_steps + 1
    env_ids  = torch.arange(env.num_envs, device=env.device)
    step_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    buffer   = torch.zeros(env.num_envs, max_len, 2, device=env.device)
    buffer[:, 0] = info["agent_pos"]

    trajectories = []
    print("Extracting trajectories...")
    while len(trajectories) < nr_runs:
        with torch.no_grad():
            action, lstm_state = agent.predict_action(obs, lstm_state, done)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        step_idx += 1
        buffer[env_ids, step_idx] = info["agent_pos"]

        if done.any():
            term_ids = terminated.nonzero(as_tuple=True)[0]
            buffer[term_ids, step_idx[term_ids]] = env.unwrapped.goal_pos
            for i in term_ids.tolist():
                trajectories.append(buffer[i, :step_idx[i] + 1].cpu().numpy().copy())

            reset_ids = done.nonzero(as_tuple=True)[0]
            obs, info = env.reset(done=done)
            buffer[reset_ids]    = 0.0
            buffer[reset_ids, 0] = info["agent_pos"][reset_ids]
            step_idx[reset_ids]  = 0

    return trajectories[:nr_runs]

def extract_trajectory_mlp(agent, env, nr_runs):

    """
     Unroll the agent (vectorized across env.num_envs parallel envs) until
     nr_runs goal-reaching trajectories have been collected. An env that gets
     truncated before reaching the goal is reset and its trajectory discarded.
     Return a list of nr_runs trajectories, each a (T_i, 2) numpy array.
    """
    obs, info = env.reset()

    max_len = env.unwrapped.max_steps + 1
    env_ids  = torch.arange(env.num_envs, device=env.device)
    step_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    buffer   = torch.zeros(env.num_envs, max_len, 2, device=env.device)
    buffer[:, 0] = info["agent_pos"]

    trajectories = []
    print("Extracting trajectories...")
    while len(trajectories) < nr_runs:
        with torch.no_grad():
            action = agent.predict_action(obs)
        obs, _, terminated, truncated, info = env.step(action)

        step_idx += 1
        buffer[env_ids, step_idx] = info["agent_pos"]

        reset_mask = terminated | truncated
        if reset_mask.any():
            term_ids = terminated.nonzero(as_tuple=True)[0]
            buffer[term_ids, step_idx[term_ids]] = env.unwrapped.goal_pos
            for i in term_ids.tolist():
                trajectories.append(buffer[i, :step_idx[i] + 1].cpu().numpy().copy())

            reset_ids = reset_mask.nonzero(as_tuple=True)[0]
            obs, info = env.reset(done=reset_mask)
            buffer[reset_ids]    = 0.0
            buffer[reset_ids, 0] = info["agent_pos"][reset_ids]
            step_idx[reset_ids]  = 0

    return trajectories[:nr_runs]

def completion_rate_recurrent(agent, env, episodes):


    num_layers = agent.backbone_model.lstm.num_layers
    hidden_size = agent.backbone_model.lstm.hidden_size
    num_envs = env.num_envs
    completed = 0
    counted = 0

    while counted < episodes:
        obs, _ = env.reset()
        # Fresh hidden state per batch — a new set of episodes must not inherit
        # the LSTM state left over from the previous batch's final step.
        lstm_state = (
                        torch.zeros((num_layers, num_envs, hidden_size),
                                            dtype=torch.float, device=env.device),
                        torch.zeros((num_layers, num_envs, hidden_size),
                                             dtype=torch.float, device=env.device)
                    )
        done    = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        active  = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        reached = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        while active.any():
            with torch.no_grad():
                action, lstm_state = agent.predict_action(obs, lstm_state, done)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            reached |= terminated & active
            active  &= ~done

        take = min(num_envs, episodes - counted)
        completed += int(reached[:take].sum().item())
        counted += take

    return (completed / episodes) * 100

def completion_rate_mlp(agent, env, episodes):

    """
    Return the episode completion rate of the trained policy in percentage.
    Episodes are rolled out in parallel across env.num_envs; batches repeat until
    at least `episodes` episodes have been scored (the last batch is trimmed so
    exactly `episodes` count toward the rate).
    """

    num_envs = env.num_envs
    completed = 0
    counted = 0

    while counted < episodes:
        obs, _ = env.reset()
        active  = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        reached = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

        while active.any():
            with torch.no_grad():
                action = agent.predict_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            reached |= terminated & active
            active  &= ~(terminated | truncated)

        take = min(num_envs, episodes - counted)
        completed += int(reached[:take].sum().item())
        counted += take

    return (completed / episodes) * 100

def _resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to exactly `n_points`, evenly spaced by arc length (endpoints kept)."""
    points = np.asarray(points, dtype=np.float64)
    n_points = max(int(n_points), 2)

    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    if total == 0.0:
        return np.repeat(points[:1], n_points, axis=0)

    s = np.linspace(0.0, total, n_points)
    return np.column_stack([np.interp(s, cum, points[:, 0]),
                            np.interp(s, cum, points[:, 1])])

def dynamic_time_warping(test_trajectory, reference_trajectory):
    """

    - The reference (a sparse waypoint polyline) is resampled to exactly as many
      points as the test trajectory, so both sequences are compared at the same
      sampling density instead of matching many dense test points against one
      far-off sparse waypoint.

    Returns -1 for an empty / degenerate input.
    """
    if len(test_trajectory) < 2 or len(reference_trajectory) < 2:
        return -1

    A = np.asarray(test_trajectory, dtype=np.float64)
    N = len(A)
    B = _resample_polyline(reference_trajectory, N)
    M = len(B)
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)  # (N, M) local costs

    D = np.zeros((N,M))
    D[0,0] = C[0,0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n,0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1,N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(
                D[n-1, m], D[n, m-1], D[n-1, m-1])

    return D[-1, -1]

def normalized_path_length(test_trajectory, reference_trajectory):

    return -1 if len(test_trajectory) == 0 else path_length(test_trajectory) / path_length(reference_trajectory)

def evaluate_model_on_metrics(agent, env, episodes, seeds,
                               nr_runs, json_path, backbone_type):

    wandb.run.define_metric("metrics/*", step_metric="metrics/seed")

    reference_trajectory = generate_reference_trajectory(json_path)
    cr_values = np.zeros((len(seeds), nr_runs))
    dtw_values = np.zeros((len(seeds), nr_runs))
    npl_values = np.zeros((len(seeds), nr_runs))

    # For Means of Means increase the nr_runs to > 1
    if backbone_type == "mlp":
        trajectory = extract_trajectory_mlp(agent, env, nr_runs)
        cr = completion_rate_mlp
    else:
        trajectory = extract_trajectory_recurrent(agent, env, nr_runs)
        cr = completion_rate_recurrent
    
    for seed in seeds:
        for i in tqdm(range(nr_runs), desc=f"Evaluating policy on metrics on seed {seed}"):
            cr_values[seed, i] = cr(agent, env, episodes)
            # Score DTW and NPL on the *same* goal-reaching rollout.
            dtw_values[seed, i] = dynamic_time_warping(trajectory[i], reference_trajectory)
            npl_values[seed, i] = normalized_path_length(trajectory[i], reference_trajectory)

            

    wandb.log({
        "metrics/seed": env.unwrapped.seed,
        "metrics/completion_rate": cr_values.mean(),
        "metrics/dynamic_time_warping":dtw_values.mean(),
        "metrics/normalized_path_length": npl_values.mean()
    })

def test_evaluate_scatter_plot(agent, env, episodes, nr_runs, json_path, backbone_type, architecture, scatter_rows):

    # scatter_rows: a list owned by the caller, shared across every architecture and seed,
    # so all results end up in one table logged once by the caller (see SCATTER_COLUMNS)

    reference_trajectory = generate_reference_trajectory(json_path)
    cr_values, dtw_values, npl_values = [], [], []

    # For Means of Means increase the nr_runs to > 1
    if backbone_type == "mlp":
        trajectory = extract_trajectory_mlp(agent, env, nr_runs)
        cr = completion_rate_mlp
    else:
        trajectory = extract_trajectory_recurrent(agent, env, nr_runs)
        cr = completion_rate_recurrent

    for i in tqdm(range(nr_runs), desc="Evaluating policy on metrics"):
        cr_values.append(cr(agent, env, episodes))
        # Score DTW and NPL on the *same* goal-reaching rollout.
        dtw_values.append(dynamic_time_warping(trajectory[i], reference_trajectory))
        npl_values.append(normalized_path_length(trajectory[i], reference_trajectory))

    cr_values = np.array(cr_values)
    dtw_values = np.array(dtw_values)
    npl_values = np.array(npl_values)

    # Append this seed's runs to the shared rows
    for c, d, n in zip(cr_values, dtw_values, npl_values):
        scatter_rows.append([architecture, env.unwrapped.seed, float(c), float(d), float(n)])

SCATTER_COLUMNS = ["architecture", "seed", "completion_rate", "dynamic_time_warping", "normalized_path_length"]

def log_scatter_plots(scatter_rows):

    # One table with every architecture/seed, plus one scatter plot per architecture and metric
    plots = {"scatter/table": wandb.Table(columns=SCATTER_COLUMNS, data=scatter_rows)}
    for architecture in dict.fromkeys(row[0] for row in scatter_rows):
        table = wandb.Table(columns=SCATTER_COLUMNS,
                            data=[row for row in scatter_rows if row[0] == architecture])
        for metric, label in [("completion_rate", "Completion rate"),
                              ("dynamic_time_warping", "DTW"),
                              ("normalized_path_length", "NPL")]:
            plots[f"scatter/{architecture}/{metric}"] = wandb.plot.scatter(
                table, "seed", metric, title=f"{label} per seed ({architecture})")

    wandb.log(plots)



def _gaussian_column(ax, x, mu, width, color, max_halfwidth=0.4):
    """Draw a vertical Gaussian N(mu, width^2) centred at x-position `x`."""
    if not np.isfinite(width) or width <= 0:
        # Zero spread (e.g. every run had CR = 1.0) -> draw a flat bar instead
        ax.hlines(mu, x - max_halfwidth, x + max_halfwidth, color=color, lw=2)
        return
    y = np.linspace(mu - 4 * width, mu + 4 * width, 200)
    pdf = norm.pdf(y, mu, width)
    w = max_halfwidth * pdf / pdf.max()  # normalise so curves never overlap
    ax.fill_betweenx(y, x - w, x + w, color=color, alpha=0.35, lw=0)
    ax.plot(x - w, y, color=color, lw=0.8)
    ax.plot(x + w, y, color=color, lw=0.8)
 
 
def plot_metrics(seeds, cr_values, dtw_values, npl_values,
                 spread="sem", show_runs=True, save_path=None):
    """
    Plot each metric as one vertical Gaussian per seed, plus the mean of means.
 
    Arrays have shape (n_seeds, nr_runs); row j belongs to seeds[j].
    spread: "sem" -> Gaussian width = uncertainty of each seed's mean (sd / sqrt(n))
            "sd"  -> Gaussian width = spread of individual runs
    Returns a dict with the mean of means and its 95% CI per metric.
    """
    metrics = {
        "Completion rate (CR)":        (cr_values,  (0.0, 1.0)),
        "Dynamic time warping (DTW)":  (dtw_values, None),
        "Normalized path length (NPL)": (npl_values, None),
    }
 
    n_seeds, n_runs = cr_values.shape
    colors = plt.cm.tab10(np.arange(n_seeds) % 10)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=(n_seeds, n_runs))
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    summary = {}
 
    for ax, (name, (vals, bounds)) in zip(axes, metrics.items()):
        means = vals.mean(axis=1)
        sds = vals.std(axis=1, ddof=1) if n_runs > 1 else np.full(n_seeds, np.nan)
        widths = sds / np.sqrt(n_runs) if spread == "sem" else sds
 
        for j in range(n_seeds):
            _gaussian_column(ax, j, means[j], widths[j], colors[j])
            if show_runs:
                ax.scatter(j + jitter[j], vals[j], s=8, color=colors[j],
                           alpha=0.6, zorder=3)
            ax.plot(j, means[j], "ko", ms=5, zorder=4)
 
        # Mean of means; CI from the spread ACROSS seeds (t-dist, few seeds)
        grand = means.mean()
        if n_seeds > 1:
            sem = means.std(ddof=1) / np.sqrt(n_seeds)
            half = t.ppf(0.975, n_seeds - 1) * sem
            ax.axhspan(grand - half, grand + half, color="crimson", alpha=0.1,
                       label="95% CI across seeds")
        else:
            half = np.nan
        ax.axhline(grand, color="crimson", lw=1.5,
                   label=f"mean of means = {grand:.3f}")
 
        if bounds is not None:
            pad = 0.05 * (bounds[1] - bounds[0])
            ax.set_ylim(bounds[0] - pad, bounds[1] + pad)
 
        ax.set_title(name)
        ax.set_xticks(range(n_seeds), [f"seed {s}" for s in seeds])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
 
        summary[name] = {"mean_of_means": grand, "ci95_halfwidth": half,
                         "per_seed_means": means}
 
    fig.suptitle(f"Per-seed Gaussians (width = {spread.upper()}, "
                 f"{n_runs} runs/seed, {n_seeds} seeds)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return summary
