#!/usr/bin/env python3
"""Optimization module"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha (float): Is the learning rate
        beta2 (float): Is the RMSProp weight
        epsilon (float): Is a small number to avoid division by zero
        var (numpy.ndarray): Is containing the variable to be updated
        grad (numpy.ndarray): Is containg the the gradient of var
        s (numpy.ndarray): Is containing the previous second moment var

    Returns:
        numpy.ndarray: The updated variable and the new moment.

    """
    Svar = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * (grad / (epsilon + np.sqrt(Svar)))
    return var, Svar
