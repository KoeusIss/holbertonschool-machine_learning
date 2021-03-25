#!/usr/bin/env python3
"""Bayesian Probabiliy module"""
import numpy as np


def factorial(k):
    """Calculates the factorial of a number

    Arguments:
        k {int} -- The given number

    Returns:
        int -- The factorial of k
    """
    fact = 1
    for i in range(1, k + 1):
        fact *= i
    return fact


def combination(n, k):
    """Calculates the combinations

    Arguments:
        n {int} -- number of trials event
        k {int} -- number of sampling size

    Returns:
        float -- The combiantion
    """
    return factorial(n) / (factorial(k) * factorial(n - k))


def intersection(x, n, P, Pr):
    """Calculates the likelihood of obtaining data given various hypothetical
    probabilities of developing severe side effects

    Arguments:
        x {int} -- Is the number of patients that develop severe side effects
        n {int} -- Is the total number of patients observed
        P {numpy.ndarray} -- Containing the various hypothetical probabilities.
        Pr {numpy.ndarray} -- Contains the prior beliefs of P

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
        numpy.ndarray: Containing the intersection of obtaining x and n.

    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any([h < 0 or h > 1 for h in P]):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any([h < 0 or h > 1 for h in Pr]):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    cmb = combination(n, x)
    L = cmb * np.power(P, x) * np.power(1 - P, n - x)
    return L * Pr
