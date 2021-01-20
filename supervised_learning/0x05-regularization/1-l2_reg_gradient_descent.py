#!/usr/bin/env python3
"""Regularization module"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    Args:
        Y (numpy.ndarray): Is contains the correct labels of the data, of shape
            (classes, m) where classes is the number of classes, and m is
            the number of data ponits.
        weights (dict): Is a dictionary of weights and biases of the neural
            network.
        cache (dict): Is a dictionary of the outputs of each layer of the
            neural network.
        alpha (float): Is the learning rate.
        lambtha (float): Is the L2 regularization parmeter.
        L (int): Is the number of layers of the network.

    Note:
        The neural network uses the `tanh` activations on each layer except
        the output layer uses `softmax`.

    """
    m = len(Y[0])
    d_z = cache["A" + str(L)] - Y
    for el in range(L, 0, -1):
        A_prev = cache["A" + str(el - 1)]
        d_W = np.matmul(d_z, A_prev.T) / m
        d_b = np.sum(d_z, axis=1, keepdims=True) / m
        d_g = 1 - np.square(A_prev)
        d_z = np.matmul(weights["W" + str(el)].T, d_z) * d_g
        reg_L2 = (1 - lambtha * alpha / m)
        weights["W" + str(el)] = reg_L2 * weights["W" + str(el)] - alpha * d_W
        weights["b" + str(el)] -= alpha * d_b
