#!/usr/bin/env python3
"""Normal module"""


class Normal:
    """Normal class represent the Normal distribution

    Note:
        If data is not given we use the given mean and stddev.

    Attributes:
        data (list): List of the data to be used to estimate the distribution
        mean (float): Is the mean of distribution
        stddev (float): Is the standard deviation of the distribution

    Raises:
        ValueError: If stddev is not positive value
        TypeError: If data is not of type list
        ValueError: If data contains less then two data points

    """
    exp = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializer"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            somme = 0
            for item in data:
                somme += (item - self.mean)**2
            self.stddev = (somme / len(data))**.5

    def z_score(self, x):
        """Calculates the z-scores of a given x-value

        Args:
            x (float): is the x-value

        Returns:
            float: the z-score of x

        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score

        Args:
            z (flaot): Is the z-score

        Returns:
            float: the x-value of z

        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF for a given x-value

        Args:
            x (float): Is the x-value

        Returns:
            float: Returns the PDF value for x

        """
        z = self.z_score(x)
        return (1 / (self.stddev * (2 * Normal.pi)**.5))\
            * Normal.exp**((-1/2) * z**2)
