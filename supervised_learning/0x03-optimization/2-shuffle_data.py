#!/usr/bin/env python3
"""Optimization module"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way

    Args:
        X (np.ndarray): Is the first array with shape (m, nx) where m is the
        number of data points and nx is the number of features in X.
        Y (np.ndarray): is the second array with shape (m, ny) where ny is the
        number of features in Y

    Returns:
        np.ndarray: The shuffled X and Y matrices

    """
    c = np.c_[X, Y]
    c = np.random.permutation(c)
    X_shuffled = c[:, :X.shape[1]]
    Y_shuffled = c[:, Y.shape[1]:]
    return X_shuffled, Y_shuffled
