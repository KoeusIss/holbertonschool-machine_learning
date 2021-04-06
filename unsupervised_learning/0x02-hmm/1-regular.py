#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def regular(P):
    """Determines the steady state of a transition matrix using linear
    algebraic method

    Arguments:
        P {np.ndarray} -- Containing the transition matrix

    Returns:
        np.ndarray -- Containing the steady state
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if np.any(np.sum(P, axis=1) != 1):
        return None
    try:
        n, _ = P.shape
        opposit = P - np.eye(n)
        opposit = np.c_[opposit, np.ones(n)]
        M = np.matmul(opposit, opposit.T)
        steady = np.linalg.solve(M, np.ones((1, n)).T).T
        if np.any(steady <= 0):
            return None
        return steady
    except Exception:
        return None
