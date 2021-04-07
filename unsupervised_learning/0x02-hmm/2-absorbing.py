#!/usr/bin/env python3
"""HMM module"""
import numpy as np


def absorbing(P):
    """Checks if a transition matrix is representing an bsorbing markov chain
    or not.

    Arguments:
        P {np.ndarray} -- containing the transition matric

    Returns:
        boolean -- True if its an absorbng markov chain, False otherwise
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if np.any(np.sum(P, axis=1) != 1):
        return False

    n, _ = P.shape
    if np.all(np.diag(P) == 1):
        return True
    if np.all(np.diag(P) != 1):
        return False

    for i in range(n - 1):
        if P[i, i] != 1 and P[i, i + 1] == 0 and P[i + 1, i] == 0:
            return False
    return True
