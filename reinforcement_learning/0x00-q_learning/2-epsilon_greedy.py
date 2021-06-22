#!/usr/bin/env python3
"""Q-learning module"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Implements Epsilon-greedy algorithm

    Arguments:
        Q {np.ndarray} -- Contains the Q-table
        state {int} -- Is the current state
        epsilon {float} -- Is the epsilon for the calculation

    Returns:
        int -- The next action index
    """
    exploration_rate_threshold = np.random.uniform()
    if exploration_rate_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(Q.shape[1])
    return action
