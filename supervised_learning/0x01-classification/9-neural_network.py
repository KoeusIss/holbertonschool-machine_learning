#!/usr/bin/env python3
"""Neural Network module"""
import numpy as np


class NeuralNetwork:
    """Neural Network class that defines a neural network with one hidden layer
    performing binary classification.

    Attributes:
        nx (int): Is the number of input features.
        nodes (int): Is the number of nodes found in the hidden layer.
        W1 (np.ndarray): Is the weight vector for the hidden layer.
        b1 (np.ndarray): Is the bias for the hidden layer.
        A1 (float): Is the activated output for the hidden layer.
        W2 (np.ndarray): Is the weight vector for the output neuron.
        b2 (float): Is the bias for the output neuron.
        A2 (float): Is the activated output for the output neuron.

    Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        TypeError: If nodes is not an integer.
        ValueError: If nodes is less than 1.

    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2
