#!/usr/bin/env python3
"""Sigma notation"""


def summation_i_squared(n):
    """Finds the summation of i squared where i goes from 1 to n

    Args:
        n (int): the stopping condition integer

    Returns:
        (int|None): The integer value of the sum, if n is not a valid number
        return None

    """
    if n < 1 or not isinstance(n, int):
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
