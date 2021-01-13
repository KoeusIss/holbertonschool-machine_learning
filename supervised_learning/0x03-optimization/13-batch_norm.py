#!/usr/bin/env python3
"""Optimization module"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
        Z (numpy.ndarray): Is the data should be normalized with shape (m, n)
            where m is the number of data points, and n is the number of
            features.
        gamma (numpy.ndarray): Is containing the scales used for batch
            normalization of shape (1, n) where n is number of features.
        beta (numpy.ndarray): Is containing the offsets used for batch
            normalization of shape (1, n) where n is the number of features.
        epsilon (float): Is a small number used to avoid division by zero.

    Returns:
        numpy.ndarray: Normalized Z matrix.

    """
    mu = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_normalized = (Z - mu) / np.sqrt(variance + epsilon)
    Z_telda = gamma * Z_normalized + beta
    return Z_telda
