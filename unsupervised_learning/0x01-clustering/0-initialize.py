#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means

    Arguments:
        X {numpy.ndarray} -- Containing the dataset
        k {int} -- Indicate the number of cluster

    Returns:
        numpy.ndarray|None -- Containing the initialized centroids for each
            dimension, Otherwise return None
    """
    if not isinstance(k, int) or k < 1:
        return None

    n, d = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return np.random.uniform(
        low=X_min,
        high=X_max,
        size=(k, d)
    )
