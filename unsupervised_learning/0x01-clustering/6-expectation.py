#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectations step in the EM algorithm

    Arguments:
        X {np.ndarray} -- Containing the data point
        pi {np.ndarray} -- Contating the priors
        m {np.ndarray} -- Containing the mean
        S {np.ndarray} -- Containing the covariance matrix

    Returns:
        np.ndarray, float -- return the posterior probability, the total
        log liklihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    nominator = np.zeros((k, n))
    for i in range(k):
        _pdf = pdf(X, m[i], S[i])
        nominator[i, :] = _pdf * pi[i]
    P = np.sum(nominator, axis=0)
    g = nominator / P
    L = np.sum(np.log(P))
    return g, L
