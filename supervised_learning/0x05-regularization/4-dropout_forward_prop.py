#!/usr/bin/env python3
"""Regularization module"""
import tensorflow as tf
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout

    Args:
        X (numpy.ndarray): Is containing the input data for the network of
            shape (nx, m) where nx is the number of input feature and m is
            the number of data points.
        weights (dict): Is containing the weights of the neural network.
        L (int): Is the number of layers in the network.
        keep_prob (float): Is the probability that a node will be kept.

    Note:
        All layers except the last should use `tanh` activation function and
        the last should use the `softmax`

    Returns:
        dict: Containing the output of each layer and the dropout mask used
            on each layer.

    """
    cache = {"A0": X}
    for el in range(1, L + 1):
        Z = np.matmul(weights["W" + str(el)], cache["A" + str(el - 1)])
        + weights["b" + str(el)]
        if el == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            cache["A" + str(el)] = A
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            cache["D" + str(el)] = D
            cache["A" + str(el)] = A / keep_prob
    return cache
