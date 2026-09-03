import numpy as np
import torch
from .geometry import path_length
import wandb

"""
Script with metric function to test the performance of trained policies

"""

MAX_TRAJECTORY_ATTEMPTS = 100  # give up looking for a goal-reaching rollout after this many tries


def _to_xy(t: torch.Tensor) -> np.ndarray:
    """(1, 2) or (2,) device tensor -> detached, copied (2,) numpy point."""
    return t.detach().reshape(-1).cpu().numpy().copy()

def extract_trajectory(agent, env):

    """
     Unroll the agent until it reaches the goal or it is truncated.
     Return the flag and the performed trajectory
    """
    obs, info = env.reset()
    done = False
    truncated = False
    trajectory = [_to_xy(info["agent_pos"])]

    while not done and not truncated:
        with torch.no_grad():
            action = agent.predict_action(obs)
        obs, _, terminated, trunc, info = env.step(action)
        done = bool(terminated.reshape(-1)[0])
        truncated = bool(trunc.reshape(-1)[0])
        trajectory.append(_to_xy(info["agent_pos"]))

    if truncated:
        return False, []
    else:
        # The environment accepts being in a radius near the goal as reaching it, so the trajectory won't have it
        # thus we need to append it to the trajectory
        trajectory.append(_to_xy(env.goal_pos))
        _, _ = env.reset()
        return done, trajectory


def completion_rate(agent, env, episodes):

    """
    Return the episodes completion rate of trained policy in percentage
    """
 
    completed_episodes = 0

    for _ in range(episodes):

        done, _ = extract_trajectory(agent, env)
        completed_episodes += done

    return (completed_episodes / episodes) * 100

def dynamic_time_warping(test_trajectory, reference_trajectory):

    if len(test_trajectory) == 0:
        return -1

    test_trajectory = np.asarray(test_trajectory, dtype=np.float64)
    reference_trajectory = np.asarray(reference_trajectory, dtype=np.float64)

    N = len(test_trajectory)
    M = len(reference_trajectory)

    D = np.zeros(shape=(N, M))

    # -- Base Cases -- #
    D[:, 0] = np.inf
    D[0,:] = np.inf 
    D[0,0] = 0.0

    for i in range(1, N):
        for j in range(1, M):
            p = test_trajectory[i]
            q = reference_trajectory[j]
            D[i,j] = np.linalg.norm(p-q) + min(
                                                D[i-1,j], 
                                                D[i, j-1],
                                                D[i-1, j-1]
                                                )

    return D[N-1,M-1]

def normalized_path_length(test_trajectory, reference_trajectory):

    return -1 if len(test_trajectory) == 0 else path_length(test_trajectory) / path_length(reference_trajectory)


def evaluate_model_on_metrics(agent, env, episodes, nr_runs, reference_trajectory = None):

    # For Means of Means increase the nr_runs to > 1
    for _ in range(nr_runs):
        cr_value = completion_rate(agent, env, episodes)

        # Score DTW and NPL on the *same* goal-reaching rollout.
        trajectory = []
        for _ in range(MAX_TRAJECTORY_ATTEMPTS):
            done, trajectory = extract_trajectory(agent, env)
            if done:
                break

        if not trajectory:
            dtw_value = npl_value = -1
        else:
            dtw_value = dynamic_time_warping(trajectory, reference_trajectory)
            npl_value = normalized_path_length(trajectory, reference_trajectory)

        wandb.log({
            "Completion Rate": cr_value,
            "Dynamic Time Warping": dtw_value,
            "Normalized Path Length": npl_value
        })



         

    






