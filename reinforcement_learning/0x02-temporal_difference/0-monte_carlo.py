#!/usr/bin/env python3
"""Temporal difference module"""


def play_episode(env, policy, max_steps):
    """Plays a single episode

    Arguments:
        env {gym.Environment} -- Is the play environment
        policy {function} -- Is the policy function
        max_steps {int} -- Is the max steps per episode

    Returns:
        list(tuple) -- Contains the result for each step
    """
    state = env.reset()
    action = policy(state)
    state_action_reward = [(state, action, None)]

    for _ in range(max_steps):
        state, reward, done, _ = env.step(action)
        action = policy(state)
        state_action_reward.append((state, action, reward))
        if done:
            break

    return state_action_reward


def monte_carlo(
    env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """Perfoms a Monte Carlo first visit algorithm

    Arguments:
        env {gym.Environment} -- The game evironment
        V {np.ndarray} -- Contains the value function
        policy {function} -- Contains the policy function

    Keyword Arguments:
        episodes {int} -- Is the number of episodes (default: {5000})
        max_steps {int} -- Is the max step per episode (default: {100})
        alpha {float} -- Exploration/exloitation rate (default: {0.1})
        gamma {float} -- Is the discount rate (default: {0.99})

    Returns:
        np.ndarray -- Updated value function
    """
    seen_state_action = set()

    for _ in range(episodes):
        state_action_reward = play_episode(env, policy, max_steps)
        T = len(state_action_reward) - 1

        G = 0
        for t in range(T - 1, -1, -1):
            state, action, _ = state_action_reward[t]
            _, _, reward_t_1 = state_action_reward[t + 1]

            G = gamma * G + reward_t_1
            if not (state, action) in seen_state_action:
                V[state] = V[state] + alpha * (G - V[state])

            seen_state_action.add((state, action))

    return V
