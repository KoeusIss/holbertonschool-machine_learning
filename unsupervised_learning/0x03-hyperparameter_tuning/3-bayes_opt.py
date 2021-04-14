#!/usr/bin/env python3
"""Hyperparameter tuning module"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization class
    """
    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True
            ):
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
