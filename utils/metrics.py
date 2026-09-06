import numpy as np
import torch
from .geometry import path_length
import wandb
from utils import generate_reference_trajectory
from tqdm import tqdm
"""
Script with metric function to test the performance of trained policies

"""

def _to_xy(t: torch.Tensor) -> np.ndarray:
    """(1, 2) or (2,) device tensor -> detached, copied (2,) numpy point."""
    return t.detach().reshape(-1).cpu().numpy().copy()

def extract_trajectory(agent, env, nr_runs):

    """
     Unroll the agent (vectorized across env.num_envs parallel envs) until
     nr_runs goal-reaching trajectories have been collected. An env that gets
     truncated before reaching the goal is reset and its trajectory discarded.
     Return a list of nr_runs trajectories, each a (T_i, 2) numpy array.
    """
    obs, info = env.reset()

    max_len = env.max_steps + 1
    env_ids  = torch.arange(env.num_envs, device=env.device)
    step_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    buffer   = torch.zeros(env.num_envs, max_len, 2, device=env.device)
    buffer[:, 0] = info["agent_pos"]

    trajectories = []
    while len(trajectories) < nr_runs:
        print("Extracting trajectories...")
        with torch.no_grad():
            action = agent.predict_action(obs)
        obs, _, terminated, truncated, info = env.step(action)

        step_idx += 1
        buffer[env_ids, step_idx] = info["agent_pos"]

        reset_mask = terminated | truncated
        if reset_mask.any():
            term_ids = terminated.nonzero(as_tuple=True)[0]
            buffer[term_ids, step_idx[term_ids]] = env.goal_pos
            for i in term_ids.tolist():
                trajectories.append(buffer[i, :step_idx[i] + 1].cpu().numpy().copy())

            reset_ids = reset_mask.nonzero(as_tuple=True)[0]
            obs, info = env.reset(done=reset_mask)
            buffer[reset_ids]    = 0.0
            buffer[reset_ids, 0] = info["agent_pos"][reset_ids]
            step_idx[reset_ids]  = 0

    return trajectories[:nr_runs]

def completion_rate(agent, env, episodes):

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


def evaluate_model_on_metrics(agent, env, eval_env, episodes, nr_runs, json_path):

    _, data = env.reset()

    start = _to_xy(data["agent_pos"][0])
    end = _to_xy(env.goal_pos)

    wandb.run.define_metric("metrics/*", step_metric="metrics/step")

    reference_trajectory = generate_reference_trajectory(json_path, start, end)

    # For Means of Means increase the nr_runs to > 1
    trajectory = extract_trajectory(agent, env, nr_runs)
    for i in tqdm(range(nr_runs), desc="Evaluating policy on metrics"):
        cr_value = completion_rate(agent, eval_env, episodes)
        # Score DTW and NPL on the *same* goal-reaching rollout.
        dtw_value = dynamic_time_warping(trajectory[i], reference_trajectory)
        npl_value = normalized_path_length(trajectory[i], reference_trajectory)

        wandb.log({
            "metrics/step": i,
            "metrics/completion_rate": cr_value,
            "metrics/dynamic_time_warping":dtw_value,
            "metrics/normalized_path_length": npl_value
        })

         

    






