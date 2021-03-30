#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    if kmax - kmin < 2:
        return None, None

    d_vars = []
    results = []
    for k in range(kmin, kmax + 1):
        k_means = kmeans(X, k)
        var = variance(X, k_means[0])
        d_vars.append(var)
        results.append(k_means)
    return results, d_vars
