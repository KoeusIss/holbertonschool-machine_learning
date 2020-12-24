#!/usr/bin/env python3
"""Neuron module"""


class Neuron:
    """Neuron class defines a single neuron performing binary classification

    Attributes
        nx (int): Is the number of input features to the neuron.
        W (numpy.ndarray): Is the weights vector for the neuron, it should be
            initialized using a random distribution.
        b (float): Is the bias for the neuron, it should initialized to 0.
        A (float): Is the activated output of the neuron (prediction),
            it should initialized to 0.


    Raises:
        TypeError: If nx is not integer
        ValueError: If nx is less then 1

    """

    def __init__(self, nx):
        """Initializer"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive')
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
