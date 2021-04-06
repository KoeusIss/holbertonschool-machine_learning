#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain being in particular state.

    Arguments:
        P {np.ndarray} -- Containing the transition matrix
        s {np.ndarray} -- Containing the the initial state

    Keyword Arguments:
        t {int} -- Is the number of iteration that the markov chain
        has been through (default: {1})

    Returns:
        np.ndarray|None -- Containing the probabilies of being in
        a specefic state
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if np.any(np.sum(P, axis=1) != 1):
        return None
    n, _ = P.shape
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if np.sum(s) != 1:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    return np.matmul(s, np.linalg.matrix_power(P, t))
