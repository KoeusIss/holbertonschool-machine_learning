#!/usr/bin/env python3
"""Q-learning module"""
import numpy as np


def q_init(env):
    """Initializes the Q-table

    Arguments:
        env {FrozenLakeEnv} -- FrozenLakeEnv instance

    Returns:
        np.ndarray -- The Q-table
    """
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))
    return qtable
