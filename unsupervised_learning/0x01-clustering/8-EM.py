#!/usr/bin/env python3
"""Clustering module"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform the Expectation maximization for the GMM

    Arguments:
        X {np.ndarray} -- Containing the data points
        k {int} -- Containing the number of class

    Keyword Arguments:
        iterations {int} -- Is the number of iterations (default: {1000})
        tol {float} -- Is the tolearance over liklihood (default: {1e-5})
        verbose {bool} -- Indicates if we could print status (default: {False})

    Returns:
        tuple(np.ndarray, float) -- the prior, the mean, the covaraince matrix
        the posterior and the Liklihood
    """
    failure = None, None, None, None, None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return failure
    if not isinstance(k, int) or k < 1:
        return failure
    if not isinstance(iterations, int) or iterations < 1:
        return failure
    if not isinstance(tol, float) or tol < 0:
        return failure
    if not isinstance(verbose, bool):
        return failure

    pi, m, S = initialize(X, k)
    g, L = expectation(X, pi, m, S)
    for it in range(iterations):
        old_L = L
        pi, m, S = maximization(X, g)
        g, L = expectation(X, pi, m, S)
        if verbose and (it % 10) == 0:
            print('Log Likelihood after {i} iterations: {l:.5f}'.format(
                i=it, l=L
            ))
        if abs(L - old_L) <= tol:
            break
    if verbose:
        print('Log Likelihood after {i} iterations: {l:.5f}'.format(
            i=it + 1, l=L
        ))
    return pi, m, S, g, L
