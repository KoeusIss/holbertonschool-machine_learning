#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters fot GMM

    Arguments:
        X {np.ndarray} -- Containing the data points

    Keyword Arguments:
        kmin {int} -- the minimum estimate number of clusters (default: {1})
        kmax {int} -- the maximum estimate number of clusters (default: {None})
        iterations {int} -- Is the number of itereations (default: {1000})
        tol {float} -- Is the tolerance allowed for liklihood (default: {1e-5})
        verbose {bool} -- Indicates the possible printing (default: {False})

    Returns:
        tuple -- the best found number of cluster, the best result,
        the liklihoods, and the BIC.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None
    if kmax - kmin < 1:
        return None, None, None, None

    n, d = X.shape
    b_lst = []
    l_lst = []
    results = []
    ks = []
    for k in range(kmin, kmax + 1):
        ks.append(k)
        em = expectation_maximization(X, k, iterations, tol, verbose)
        if em is (None, None, None, None, None):
            return None, None, None, None
        pi, m, S, g, L = em
        results.append((pi, m, S))
        p = k * d + (k - 1) + k * d * (d + 1) / 2
        b_lst.append(p * np.log(X.shape[0]) - 2 * L)
        l_lst.append(L)

    bics = np.array(b_lst)
    liklihoods = np.array(l_lst)
    best_idx = np.argmin(bics)
    best_k = ks[best_idx]
    best_result = results[best_idx]
    return best_k, best_result, liklihoods, bics
