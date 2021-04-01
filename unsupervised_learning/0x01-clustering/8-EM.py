#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the EM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, L = expectation(X, pi, m, S)
    for it in range(iterations):
        old_L = L
        if verbose and (it % 10) == 0:
            print('Log Likelihood after {i} iterations: {l:.5f}'.format(
                i=it, l=L
            ))
        pi, m, S = maximization(X, g)
        g, L = expectation(X, pi, m, S)
        if abs(L - old_L) <= tol:
            break
    if verbose:
        print('Log Likelihood after {i} iterations: {l:.5f}'.format(
            i=it + 1, l=L
        ))
    return pi, m, S, g, L
