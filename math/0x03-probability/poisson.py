#!/usr/bin/env python3
"""Piosson module"""


class Poisson:
    """Poisson class represent the poisson distribution

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
        self.data = data
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of PMF for a given number of ``successes``

        Args:
            k (int): Is the number of successes

        Returns:
            float|0: The PMF value for k, 0 if k is out of range

        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            return (self.lambtha**k * Poisson.exp**((-1) * self.lambtha))\
                / Poisson.factorial(k)

    def cdf(self, k):
        """Calculates the value of CDF for a given number of ``successes``

        Args:
            k (int): Is the given number of successes

        Returns:
            float|0: The CDF value for k, 0 if k is out of range

        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        somme = 0
        for i in range(0, k+1):
            somme += self.lambtha**i / Poisson.factorial(i)
        return Poisson.exp**((-1) * self.lambtha) * somme

    @staticmethod
    def factorial(k):
        """Calculates the factorial of a given value

        Args:
            k (int): the given integer

        Returns:
            (int): the factoriam of the given k integer

        """
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return fact
