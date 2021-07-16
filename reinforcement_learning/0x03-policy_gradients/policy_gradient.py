#!/usr/bin/env python3
"""Policy Gradient module"""
import numpy as np


def softmax(X):
    """Computes the softmax of X.

    Arguments:
        X {np.ndarray} -- Contains the input array.

    Returns:
        np.ndarray -- Contains the softmax of X.
    """
    ex = np.exp(X - np.max(X))
    return ex / np.sum(ex)


def policy(matrix, weight):
    """Computes the weighted policy.

    Arguments:
        matrix {np.ndarray} -- Contains the given matrix.
        weight {np.ndarray} -- Contains the weights.

    Returns:
        np.ndarray -- Contains the weighted matrix.
    """
    X = matrix @ weight
    return softmax(X)


def softmax_grad(softmax):
    """Computes the gradient of a given softmax

    Arguments:
        softmax {np.ndarray} -- Contains the softmax matrix.

    Returns:
        np.ndarray -- The gradient of the softmax
    """
    softmax = softmax.reshape(-1, 1)
    return np.diagflat(softmax) - softmax @ softmax.T


def policy_gradient(state, weight):
    """Computes the Monte Carlo weighted policy gradient.

    Arguments:
        state {np.ndarray} -- Contains the states.
        weight {np.ndarray} -- Contains Theta weights.

    Returns:
        tuple(np.ndarray) -- Contains the action and the gradient.
    """
    PI = policy(state, weight)
    action = np.random.choice(len(PI[0]), p=PI[0])

    d_PI = softmax_grad(PI)[action, :]
    d_log = d_PI / PI[0, action]
    grad = state.T @ d_log[None, :]
    return action, grad
