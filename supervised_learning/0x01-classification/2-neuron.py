#!/usr/bin/env python3
"""Neuron module"""
import numpy as np


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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Is the input data  with sahpe (nx, m) where
            nx is the number of input features to the neuron and m is the
            number of training examples

        Returns:
            (float): Returns the private activation output self.__A

        """
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp((-1) * z))
        self.__A = sigmoid
        return self.__A
