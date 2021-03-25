#!/usr/bin/env python3
"""Bayesian Probabiliy module"""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data.

    Arguments:
        x {int} -- Is the number of patients that develop severe side effects
        n {int} -- Is the total number of patients observed
        p1 {float} -- Is the lower bound of the range.
        p2 {float} -- Is the upper bound of the range.

    Raises:
        ValueError: If n not a positive integer
        ValueError: If x not a positive integer
        ValueError: If x grater than n
        TypeError: If P not a 1D numpu.ndarray
        TypeError: If Pr not a numpy.ndarray with the same shape of P
        ValueError: If any of P items not in range [0, 1]
        ValueError: If any of Pr items not in range [0, 1]
        ValueError: If the sum of Pr items different than 1

    Returns:
        numpy.ndarray: posterior probability of each P.

    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    beta_1 = special.betainc(x + 1, n - x + 1, p1)
    beta_2 = special.betainc(x + 1, n - x + 1, p2)
    return beta_2 - beta_1
