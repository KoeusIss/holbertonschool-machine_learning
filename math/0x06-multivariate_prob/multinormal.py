#!/usr/bin/env python3
"""Multivariate probability module"""
import numpy as np


class MultiNormal:
    """
    MultiNormal class

    Attriburte:
        data (numpy.ndarray): Of shape (d, n) containing the data set where
            n is the number of data points, and d is the number of dimensions.

    Raises:
        TypeError: If data is not a 2D numpy.ndarray.
        ValueError: If n is less than 2

    """
    def __init__(self, data):
        """
        Consturctor
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """Calculates the mean and covariance of data set

        Args:
            X (np.ndarray): of shape (n, d) containing the data set where n is
                the number of data point, and d is the number of dimension.

        Returns:
            np.ndarray: of shape (1, d) represent the mean
            np.ndarray: of shape (d, d) represent the covariance matrix

        """
        d, n = X.shape
        mean = np.expand_dims(np.mean(X, axis=1), axis=1)
        cov = np.matmul((X - mean), (X - mean).T) / (n - 1)
        return mean, cov
