#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM

    Arguments:
        X {np.ndarray} -- Containing the data points
        g {np.ndarray} -- Containing the posterior probabilities

    Returns:
        tuple(np.ndarray) -- Contating the prior, the mean and the covariance
        matrix respectively
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if g.shape[1] != X.shape[0]:
        return None, None, None
    if np.any(np.sum(g, axis=0) != np.ones((g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        pi[i] = np.average(g[i], axis=0)
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / np.sum(g[i], axis=0)
        S[i] = np.matmul(g[i] * (X - m[i]).T, X - m[i]) / np.sum(g[i], axis=0)
    return pi, m, S
