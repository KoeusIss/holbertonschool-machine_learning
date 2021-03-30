#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for Guassian Mixture Model

    Arguments:
        X {np.ndarray} -- Containing the dataset
        k {int} -- is the number of cluster

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray) -- Returns the priors for
        each cluster, the centroid initialized, covariance matrix.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape
    pi = (1 / k) * np.ones(k)
    m, _ = kmeans(X, k)
    S = np.ones((k, d, d)) * np.eye(d)
    return pi, m, S
