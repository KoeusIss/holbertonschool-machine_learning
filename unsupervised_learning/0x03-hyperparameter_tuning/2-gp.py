#!/usr/bin/env python3
"""Hyperparameter tuning module"""
import numpy as np


class GaussianProcess:
    """GaussianProcess class
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initializer

        Arguments:
            X_init {np.ndarray} -- Contains the inputs already sampled with
            the black-box function of shape (t, 1)
            Y_init {np.ndarray} -- Contains the outputs of the black-box
            function for each input X_init of shae (t, 1)

        Keyword Arguments:
            l {int} -- Is the length parameter of kernel (default: {1})
            sigma_f {float} -- Is the standard deviation (default: {1})
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Computes the Isotropic squared exponential Kernel

        Arguments:
            X1 {np.ndarray} -- Contains the sampled inputs of shape (m, 1)
            X2 {np.ndarray} -- Contains the sampled star input of shape (n, 1)

        Returns:
            np.ndarray -- The covariance matrix of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1)\
            + np.sum(X2**2, axis=1) - 2 * np.matmul(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and the standard deviation of points in a Gaussian
        process

        Arguments:
            X_s {np.ndarray} -- Containing all the points whose mean and std
            deviation should be calculated

        Returns:
            tuple(np.ndarrat) -- mu, sigma
        """
        s, _ = X_s.shape

        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(s,)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates the Gaussian processs

        Arguments:
            X_new {np.ndarray} -- The new sample point
            Y_new {np.ndarray} -- The new sample function output
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
