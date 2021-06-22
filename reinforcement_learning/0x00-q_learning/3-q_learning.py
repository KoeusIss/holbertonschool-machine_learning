#!/usr/bin/env python3
"""Q-learning module"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """Performs Q-learning

    Arguments:
        env {FrozenLakeEnv} -- FrozenLakeEnv instance
        Q {np.ndarray} -- Contains the Q-table

    Keyword Arguments:
        episodes {int} -- The total number of episodes (default: {5000})
        max_steps {int} -- The max allowed steps (default: {100})
        alpha {float} -- Is the learning rate (default: {0.1})
        gamma {float} -- Is the discount rate (default: {0.99})
        epsilon {int} -- Is the initial threshold (default: {1})
        min_epsilon {float} -- min epsilon value (default: {0.1})
        epsilon_decay {float} -- Is epsilon decay Rate (default: {0.05})

    Returns:
        tuple(np.ndarray, list) -- The Q-table, and the list of the rewards
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _ = env.step(action)

            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break
        epsilon = min_epsilon + \
            (epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
