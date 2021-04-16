#!/usr/bin/env python3
"""Hyperparameter tuning module"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization class
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializer

        Arguments:
            f {function} -- Is the black-box function to optimized
            X_init {np.ndarray} -- Contains the inputs already sampled
            Y_init {np.ndarray} -- Contains the outputs of the black-box
            bounds {tuple} -- Contains the bounds of the space
            ac_samples {int} -- Is the number of samples that should be
            analysed

        Keyword Arguments:
            l {int} -- length parameter of the kernel (default: {1})
            sigma_f {float} -- Is the standard deviation (default: {1})
            xsi {float} -- Is the exploration-exploitation (default: {0.01})
            minimize {bool} -- Minimization vs Maximization (default: {True})
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location

        Returns:
            tuple -- Contains the next best sample, and the expected
            improvement matrix
        """
        mu, sigma = self.gp.predict(self.X_s)

        with np.errstate(divide='warn'):
            imp = mu - self.xsi - np.min(self.gp.Y)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        samples = []
        for _ in range(iterations):
            X_next, ei = self.acquisition()
            if X_next in samples:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            samples.append(X_next)

        return X_next, Y_next
