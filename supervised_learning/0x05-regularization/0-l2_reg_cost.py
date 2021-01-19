#!/usr/bin/env python3
"""Regularization module"""
import tensorflow as tf
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
    norm = np.array(())
    for weight in weights.values():
        norm = np.append(norm, np.linalg.norm(weight, "fro"))
    return cost + lambtha / (2 * m) * np.sum(norm)
