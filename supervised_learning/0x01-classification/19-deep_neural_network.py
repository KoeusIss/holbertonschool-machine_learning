#!/usr/bin/env python3
"""Deep Neural Network module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeural Network class that defines a neural network performing binary
    classification.

    Attributes:
        nx (int): Is the number of input features.
        layers (list): Is the list contains the lenght of the network layers
        L (int): the number of layers in the neural network
        cache (dict): Holds all the intermediary values of the network
        weights (dict): Holds all the weights and biased of the network.

    Raises:
        TypeError: If nx is not an integer
        ValueError: If nx less than 1
        TypeError: If layers is not a list on positive integer

    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = dict()
        lst = layers.copy()
        lst.insert(0, self.nx)
        for l in range(1, self.L + 1):
            self.__weights['W' + str(l)] = np.random.randn(
                lst[l], lst[l - 1]) * np.sqrt(2 / lst[l - 1])
            self.__weights['b' + str(l)] = np.zeros((lst[l], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Is the input data with shape (nx, m) where
            nx is the number of input features of the neuron and m the number
            of examples.

        Returns:
            (numpy.ndarray): The output of the neural network.

        """
        self.__cache["A0"] = X
        for l in range(1, self.L + 1):
            z_tmp = np.matmul(self.weights["W" + str(l)],
                              self.__cache["A" + str(l - 1)]) + self.weights[
                                  "b" + str(l)]
            A_tmp = 1 / (1 + np.exp((-1) * z_tmp))
            self.__cache["A" + str(l)] = A_tmp
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Is the correct labels for the input data
            with shape (1, m) where m is the number of example
            A (numpy.ndarray): Is the activated output of the neuron for each
            example.

        Returns:
            float: Returns the cost.

        """
        m = len(Y[0])
        J = (-1/m) * (np.matmul(np.log(A), Y.T)
                      + np.matmul(np.log(1.0000001 - A), (1 - Y).T))
        return np.sum(J)
