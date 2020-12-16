#!/usr/bin/env python3
"""Binomial module"""


class Binomial:
    """Binomial class represent the Binomial distribution

    Note:
        If data is not given we use the given n and p.

    Attributes:
        data (list): List of the data to be used to estimate the distribution
        n (float): Is the number of Bernoulli trials
        p (float): Is the probability of a `success`

    Raises:
        ValueError: If stddev is not positive value
        TypeError: If data is not of type list
        ValueError: If data contains less then two data points

    """
    exp = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """Initializer"""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            somme = 0
            for item in data:
                somme += (item - mean)**2
            variance = somme / len(data)
            self.p = 1 - variance / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the PMF of the given number of success

        Args:
            k (int): Is the given number of success

        Returns:
            float|0: Returns the PMF value for x, or 0 if k is out of range

        """
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        return Binomial.combination(self.n, k)\
            * self.p**k * (1 - self.p)**(self.n - k)

    @staticmethod
    def factorial(k):
        """Calculates the factorial of a given value

        Args:
            k (int): the given integer k
        Returns:
            (int): the factoriam of the given k integer

        """
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return fact

    @staticmethod
    def combination(n, k):
        """Calculates the binomial coefficient for n choose k

        Args:
            n (int): the geiven integer n
            k (int): the given integer k
        Returns:
            (int): Returns the binomial coefficient

        """
        return Binomial.factorial(n)\
            / (Binomial.factorial(k) * Binomial.factorial(n - k))
