#!/usr/bin/env python3
"""Optimization module"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with momentum optimization
    algorithm

    Args:
        alpha (float): Is the learning rate.
        beta1 (float): Is the momentum weight hyperparameter.
        var (numpy.ndarray): Is containing the variable to be updated.
        grad (numpy.ndarray): Is containing the gradient of var
        v (numpy.ndarra): Is the previous first of var

    Returns:
        numpy.ndarray: the updated variable and the new moment.

    """
    Vvar = beta1 * v + (1 - beta1) * grad
    var = var - alpha * Vvar
    return var, Vvar
