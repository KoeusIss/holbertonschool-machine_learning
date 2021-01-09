#!/usr/bin/env python3
"""Optimization module"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constant of matrix

    Args:
        X (np.ndarray): Is the input matrix with shape (m, nx) where m is
        the number of data point and nx the number of feature.

    Returns:
        np.ndarray: The mean and the standard deviation of each feature.

    """
    mean = np.mean(X, 0)
    stddev = np.std(X, 0)
    return mean, stddev
