#!/usr/bin/env python3
"""Policy Gradient module"""
import numpy as np
from policy_gradient import policy
from policy_gradient import policy_gradient


def play_episode(env, weight):
    """Plays a single episode

    Arguments:
        env {gym.Environment} -- Is the play environment
        weight {function} -- Is the policy function
        max_steps {int} -- Is the max steps per episode

    Returns:
        list(tuple) -- Contains the result for each step
    """
    state = env.reset()[None, :]
    state_action_reward_grad = []

    while True:
        action, grad = policy_gradient(state, weight)
        state, reward, done, _ = env.step(action)
        state = state[None, :]
        state_action_reward_grad.append((state, action, reward, grad))
        if done:
            break
    return state_action_reward_grad


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Trains a Policy Gradient REINFORCE model

    Arguments:
        env {gym.Environment} -- is the play environment.
        nb_episodes {int} -- The number of episodes.

    Keyword Arguments:
        alpha {float} -- Is the learning rate (default: {0.000045})
        gamma {float} -- Is the discount factor (default: {0.98})

    Returns:
        list -- The scores list during episodes
    """
    weight = np.random.rand(4, 2)
    rewards = []

    for episode in range(nb_episodes):
        state_action_reward_grad = play_episode(env, weight)
        T = len(state_action_reward_grad) - 1

        score = 0
        for t in range(0, T):
            _, _, reward, grad = state_action_reward_grad[t]
            score += reward

            G = np.sum([
                gamma**(k - t - 1) * state_action_reward_grad[k][2] for k in
                range(t + 1, T + 1)])
            weight += alpha * G * gamma**t * grad
        rewards.append(score)
        print("{}: {}".format(episode, score), end="\r", flush=False)
    return rewards
