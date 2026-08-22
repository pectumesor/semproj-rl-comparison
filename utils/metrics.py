import numpy as np
import math

"""
Script with metric function to test the performance of trained policies

"""

def completion_rate(agent, env, episodes):

    """
    Return the episodes completion rate of trained policy in percentage
    """
 
    completed_episodes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while(not done or not truncated):
            action = agent.predict_action(obs)
            obs, _, done, truncated, _ = env.step(action)

        completed_episodes += done is True

    return (completed_episodes / episodes) * 100

def dynamic_time_warping(test_trajectory, reference_trajectory):

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

    return D[N,M]
