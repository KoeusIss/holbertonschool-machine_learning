#!/usr/bin/env python3
"""Optimization module"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix

    Args:
        X (np.ndarray): Is a matrix to normalize of shape (d, nx) where d is
        the number of data points and nx is the number of features
        m (np.ndarray): Is the mean of all features of X with shape (nx, )
        s (np.ndarray): Is the standard deviation of all features of X with
        shapr (nx, ).

    Returns:
        np.ndarray: The normalized version of X

    """
    return (X - m) / s
