#!/usr/bin/env python3
"""Regularization module"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y (numpy.ndarray): Is containing the correct labels for the data of
            shape (classes, m) where classes is the number of classes and m
            is the number of data points.
        weights (dict): Is a dictionary of the weight and biases of the network
        cache (dict): Is a dictionary of the ouputs and dopout masks of each
            layer of the neural network.
        alpha (float): Is the learning rate.
        keep_probs (float): Is the probability that a node will be kept.
        L (int): Is the number of layers of the network.

    Note:
        All layers should use the `tanh` activation function, except the last
        one using `softmax`.

    """
    m = len(Y[0])
    d_z = cache["A" + str(L)] - Y
    for el in range(L, 0, -1):
        A_prev = cache["A" + str(el - 1)]
        d_W = np.matmul(d_z, A_prev.T) / m
        d_b = np.sum(d_z, axis=1, keepdims=True) / m
        d_g = 1 - np.square(A_prev)
        d_A = np.matmul(weights["W" + str(el)].T, d_z)
        if el > 1:
            d_A = d_A * cache["D" + str(el - 1)]
            d_A = d_A / keep_prob
        d_z = d_A * d_g
        weights["W" + str(el)] -= alpha * d_W
        weights["b" + str(el)] -= alpha * d_b
