#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance

    Arguments:
        X {np.ndarray} -- Containing the dataset
        C {np.ndarray} -- Containing the centroids

    Returns:
        float -- The intra-cluster variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    distances = np.sqrt(np.sum((X - C[:, np.newaxis])**2, axis=2))
    return np.sum(np.min(distances, axis=0)**2)
