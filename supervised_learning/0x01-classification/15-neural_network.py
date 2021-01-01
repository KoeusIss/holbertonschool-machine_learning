#!/usr/bin/env python3
"""Neural Network module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates the forward propgation of the neural network

        Args:
            X (np.ndarray): Is the input data with shape (nx, m) where nx is
            the number of input features to the neuron, and m is the number of
            examples.

        Returns:
            (np.ndarray): Returns the activated output for the hidden layers
            (np.ndarray): Returns the activated output for the output neuron

        """
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp((-1) * z1))
        z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp((-1) * z2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (np.ndarray): Is the correct labels for the input data with
            shape (1, m) where m is the number of examples
            A (np.ndarray): the activated output of the neuron for each

        Returns:
            (float): the cost

        """
        m = len(Y[0])
        J = (-1/m) * (np.matmul(np.log(A), Y.T)
                      + np.matmul(np.log(1.0000001 - A), (1 - Y).T))
        return np.sum(J)

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X (np.ndarray): Is the input data
            Y (np.ndarray): Is the correct label for the input data

        Returns:
            (numpy.ndarray): The predicted labels for each example
            (float): the cost of the predicted values

        """
        self.forward_prop(X)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            X (numpy.ndarray): Is the input data
            Y (numpy.ndarray): Is the correct labels for the input data
            A1 (numpy.ndarray): Is the output of the hidden layer
            A2 (numpy.ndarray): Is the predicted output
            alpha (float): Is the learning rate

        """
        m = Y.shape[1]
        d_z2 = A2 - Y
        d_sig = A1 * (1 - A1)
        d_z1 = np.matmul(self.W2.T, d_z2) * d_sig
        self.__W1 -= alpha * np.matmul(d_z1, X.T) / m
        self.__b1 -= alpha * np.sum(d_z1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * np.matmul(d_z2, A1.T) / m
        self.__b2 -= alpha * np.sum(d_z2, axis=1, keepdims=True) / m

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neural network

        Args:
            X (numpy.ndarray): Is the input data
            Y (numpy.ndarray): Is the correct labels for the input
            iterations (int): Is the number of iterations to train over
            alpha (float): Is the learning rate
            verbose (boolean): Defines whether or not print information about
            training.
            graph (boolean): Defines whether or not to graph information about
            training once the iterations occured.
            step (int): The steps number

        Raises:
            TypeError: If iterations is not an iteger
            ValueError: If iterations is not positive integer
            TypeError: If alpha is not a float number
            ValueError: If alpha is not a positve float

        Returns:
            float: The evaluation of the training data after iterations occured

        """
        costs = np.array(())
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step not in range(0, iterations):
                raise ValueError('step must be positive and <= iterations')
        for iteration in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
            if verbose and step in range(0, iterations + 1):
                cost = self.cost(Y, self.A2)
                costs = np.append(costs, cost)
                print(f"Cost after {iteration} iterations: {cost}")
        if graph:
            plt.plot(costs, 'b-')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)