#!/usr/bin/env python3
"""Optimization module"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha (float): Is the learning rate
        beta1 (float): Is the weight used for first moment
        beta2 (float): Is the weight usef for second moment
        epsilon (float): Is small number to avoid division by zero
        var (numpy.ndarray): Is containing the variable to be updated
        grad (numpy.ndarray): Is containing the gradient of var
        v (numpy.ndarray): Is the previous first moment
        s (numpy.ndarray): Is the previous second moment
        t (float): Is the time step used for bias correction

    Returns:
        numpy.ndarray: The updated variables, the new first moment and
            the second moment.

    """
    Vvar = beta1 * v + (1 - beta1) * grad
    Svar = beta2 * s + (1 - beta2) * grad**2
    Vvar_corrected = Vvar / (1 - beta1**t)
    Svar_corrected = Svar / (1 - beta2**t)
    var = var - alpha * (Vvar_corrected / (epsilon + np.sqrt(Svar_corrected)))
    return var, Vvar, Svar
