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
