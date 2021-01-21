#!/usr/bin/env python3
"""Regularization module"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Claculates the cost of a neural network with L2 regularization

    Args:
        cost (numpy.ndarray): Is the cost of newtwork without L2 regularization
        lambtha (float): Is the regularization parameter.
        weights (dict(numpy.ndarray)): Is the dictionary of weights and biases
            of the neural network.
        L (int): Is the number of layers in the neural network.
        m (int): Is the number of data points used.

    Returns:
        numpy.ndarray: The cost of network acounting for L2 regularization.

    """
    norm = 0
    for el in range(1, L + 1):
        weight = weights["W" + str(el)]
        norm += np.linalg.norm(weight, "fro")
    return cost + lambtha / (2 * m) * norm
