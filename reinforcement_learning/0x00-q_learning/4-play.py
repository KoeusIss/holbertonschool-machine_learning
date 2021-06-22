#!/usr/bin/env python3
"""Q-learning module"""
import numpy as np


def play(env, Q, max_steps=100):
    """Plays an episode

    Arguments:
        env {FrozenLakeEnv} -- FrozenLakeEnv instance
        Q {np.ndarray} -- Contains the Q-table

    Keyword Arguments:
        max_steps {int} -- The maximum steps per episode (default: {100})

    Returns:
        float -- The toral rewards for the episode
    """
    state = env.reset()
    done = False

    for _ in range(max_steps):
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)

        if done:
            env.render()
            break
        state = new_state
    env.close()
    return reward
