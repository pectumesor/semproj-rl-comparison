import warnings
from pathlib import Path

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
    # Each iteration is one env step, so after a few horizons every env has finished several
    # episodes; a policy that still hasn't reached the goal by then is not going to
    cutoff = 0
    while len(trajectories) < nr_runs and cutoff < 3 * max_len:
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
        cutoff += 1

    nr_traj_found = min(len(trajectories), nr_runs) # will be == nr_runs if we succeed, otherwise less if cutt off

    return trajectories[:nr_traj_found], nr_traj_found

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
    # Each iteration is one env step, so after a few horizons every env has finished several
    # episodes; a policy that still hasn't reached the goal by then is not going to
    cutoff = 0
    while len(trajectories) < nr_runs and cutoff < 3 * max_len:
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
        cutoff+=1

    nr_traj_found = min(len(trajectories), nr_runs) # will be == nr_runs if we succeed, otherwise less if cutt off

    return trajectories[:nr_traj_found], nr_traj_found

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

        # Only the first `take` envs are scored, so stop once those have finished
        take = min(num_envs, episodes - counted)
        while active[:take].any():
            with torch.no_grad():
                action, lstm_state = agent.predict_action(obs, lstm_state, done)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            reached |= terminated & active
            active  &= ~done

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

        # Only the first `take` envs are scored, so stop once those have finished
        take = min(num_envs, episodes - counted)
        while active[:take].any():
            with torch.no_grad():
                action = agent.predict_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            reached |= terminated & active
            active  &= ~(terminated | truncated)

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

def evaluate_model_on_metrics(agent, env, episodes,
                               nr_runs, json_path, storage_path, backbone_type):
    """
    Score one trained seed and save its raw per-run values to storage_path as
    {cr,dtw,npl}_values.npy. plot_mean_of_means later aggregates the seed folders.
    """

    reference_trajectory = generate_reference_trajectory(json_path)

    # For Means of Means increase the nr_runs to > 1
    if backbone_type == "mlp":
        trajectory, nr_traj_found = extract_trajectory_mlp(agent, env, nr_runs)
        cr = completion_rate_mlp
    else:
        trajectory, nr_traj_found = extract_trajectory_recurrent(agent, env, nr_runs)
        cr = completion_rate_recurrent

    if nr_traj_found == 0: # Untrained/bad policy, no successful trajectories found
        # CR is genuinely 0. DTW and NPL are undefined without a trajectory, and 0 would read as a
        # perfect score, so they are saved empty and plot_mean_of_means leaves this seed out of them
        cr_values = np.zeros(nr_runs)
        dtw_values = np.array([])
        npl_values = np.array([])

    else:
        # CR rolls out its own episodes; DTW and NPL score the goal-reaching rollouts found,
        # which can be fewer than nr_runs if the extraction hit its cutoff
        cr_values = np.array([cr(agent, env, episodes)
                              for _ in tqdm(range(nr_runs), desc="Evaluating completion rate")])
        dtw_values = np.array([dynamic_time_warping(traj, reference_trajectory) for traj in trajectory])
        npl_values = np.array([normalized_path_length(traj, reference_trajectory) for traj in trajectory])

    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    np.save(storage_path / "cr_values.npy", cr_values)
    np.save(storage_path / "dtw_values.npy", dtw_values)
    np.save(storage_path / "npl_values.npy", npl_values)

# (file prefix, plot title, y-limits). CR is a percentage, see completion_rate_mlp
MEAN_OF_MEANS_METRICS = [
    ("cr",  "Completion rate (CR) [%]",     (0.0, 100.0)),
    ("dtw", "Dynamic time warping (DTW)",   None),
    ("npl", "Normalized path length (NPL)", None),
]

def load_seed_metrics(metric_dir):
    """
    Return {seed: {"cr": values, "dtw": values, "npl": values}} for every seed_<s> folder
    in metric_dir holding all three arrays. Seeds whose job died before saving are skipped.
    """
    per_seed = {}
    seed_dirs = sorted(Path(metric_dir).glob("seed_*"), key=lambda p: int(p.name.split("_")[-1]))
    for seed_dir in seed_dirs:
        paths = {key: seed_dir / f"{key}_values.npy" for key, _, _ in MEAN_OF_MEANS_METRICS}
        if not all(path.is_file() for path in paths.values()):
            warnings.warn(f"Incomplete metrics in {seed_dir}, skipping it")
            continue
        per_seed[int(seed_dir.name.split("_")[-1])] = {key: np.load(path) for key, path in paths.items()}
    return per_seed

def _gaussian_column(ax, x, mu, width, color, max_halfwidth=0.4):
    """Draw a vertical Gaussian N(mu, width^2) centred at x-position `x`."""
    if not np.isfinite(width) or width <= 0:
        # Zero spread (e.g. every run had CR = 100) or a single run -> draw a flat bar instead
        ax.hlines(mu, x - max_halfwidth, x + max_halfwidth, color=color, lw=2)
        return
    y = np.linspace(mu - 4 * width, mu + 4 * width, 200)
    pdf = norm.pdf(y, mu, width)
    w = max_halfwidth * pdf / pdf.max()  # normalise so curves never overlap
    ax.fill_betweenx(y, x - w, x + w, color=color, alpha=0.35, lw=0)
    ax.plot(x - w, y, color=color, lw=0.8)
    ax.plot(x + w, y, color=color, lw=0.8)

def plot_mean_of_means(metric_dir, title="", spread="sem", show_runs=True):
    """
    Plot each metric as one vertical Gaussian per seed, plus the mean of the per-seed means
    with its 95% CI across seeds (t-distribution, since there are few seeds).

    metric_dir: folder with one seed_<s>/ subfolder per seed, as written by evaluate_model_on_metrics
    spread: "sem" -> Gaussian width = uncertainty of each seed's mean (sd / sqrt(n))
            "sd"  -> Gaussian width = spread of individual runs
    Returns (figure, summary), summary holding the mean of means and CI half-width per metric,
    or (None, None) if metric_dir has no complete seed.
    """
    per_seed = load_seed_metrics(metric_dir)
    if not per_seed:
        return None, None

    seeds = list(per_seed)
    n_seeds = len(seeds)
    seed_colors = dict(zip(seeds, plt.cm.tab10(np.arange(n_seeds) % 10)))  # same colour per seed in every panel
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    summary = {"seeds": seeds}

    for ax, (key, name, bounds) in zip(axes, MEAN_OF_MEANS_METRICS):
        # Seeds can hold different numbers of runs (DTW/NPL only score the trajectories found),
        # and none at all for DTW/NPL when the policy never reached the goal
        metric_seeds = [seed for seed in seeds if len(per_seed[seed][key]) > 0]
        runs = [per_seed[seed][key] for seed in metric_seeds]
        means = np.array([vals.mean() for vals in runs])
        n_metric_seeds = len(metric_seeds)

        if n_metric_seeds == 0:
            ax.text(0.5, 0.5, "no goal-reaching trajectories", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            summary[key] = {"mean_of_means": float("nan"), "ci95_halfwidth": float("nan"),
                            "per_seed_means": means}
            continue

        for j, vals in enumerate(runs):
            sd = vals.std(ddof=1) if len(vals) > 1 else np.nan
            width = sd / np.sqrt(len(vals)) if spread == "sem" else sd
            _gaussian_column(ax, j, means[j], width, seed_colors[metric_seeds[j]])
            if show_runs:
                ax.scatter(j + rng.uniform(-0.08, 0.08, size=len(vals)), vals, s=8,
                           color=seed_colors[metric_seeds[j]], alpha=0.6, zorder=3)
            ax.plot(j, means[j], "ko", ms=5, zorder=4)

        # Mean of means; CI from the spread ACROSS seeds
        grand = means.mean()
        if n_metric_seeds > 1:
            sem = means.std(ddof=1) / np.sqrt(n_metric_seeds)
            half = t.ppf(0.975, n_metric_seeds - 1) * sem
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
        ax.set_xticks(range(n_metric_seeds), [f"seed {s}" for s in metric_seeds])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        summary[key] = {"mean_of_means": float(grand), "ci95_halfwidth": float(half),
                        "per_seed_means": means}

    fig.suptitle(f"{title}\nPer-seed Gaussians (width = {spread.upper()}, {n_seeds} seeds)")
    fig.tight_layout()
    return fig, summary
