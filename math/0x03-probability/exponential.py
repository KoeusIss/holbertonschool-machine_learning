#!/usr/bin/env python3
"""Piosson module"""


class Exponential:
    """Exponential class represent the Exponential distribution

    Note:
        If data is not given we use the given lambtha.

    Attributes:
        data (list): List of the data to be used to estimate the distribution
        lambtha (float): Expected number of occurences in a given time frame

    Raises:
        ValueError: If lambtha is not positive value
        TypeError: If data is not of type list
        ValueError: If data contains less then two data points

    """
    exp = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initializer"""
        if not data:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period

        Args:
            x (float|int): Is the given time period

        Returns:
            float|0: Returns the PDF value for x, If x out of range return 0

        """
        if x < 0:
            return 0
        return self.lambtha * Exponential.exp**((-1) * self.lambtha * x)
