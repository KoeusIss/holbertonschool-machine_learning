#!/usr/bin/env python3
"""Neuron module"""
import numpy as np
import matplotlib.pyplot as plt


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
            numpy.ndarray: Returns the private activation output self.__A

        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + 1 / np.exp(z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Is the correct label for the input data with
            shape (1, m) where m is the number of examples
            A (numpy.ndarray): Is the activated output of the neuron with
            shape (1, m) where m is the number of examples

        Returns:
            float: Returns the cost

        """
        m = len(Y[0])
        J = (-1/m) * (np.matmul(np.log(A), Y.T)
                      + np.matmul(np.log(1.0000001 - A), (1 - Y).T))
        return np.sum(J)

    def evaluate(self, X, Y):
        """Evaluates the neuron predictions

        Args:
            X (numpy.ndarray): Is the input data
            Y (numpu.ndarray): Is the correct labels for the input data

        Returns:
            numpy.ndarray: the neuron prediction
            float: the cost

        """
        self.forward_prop(X)
        prediction = np.where(self.A >= 0.5, 1, 0)
        return prediction, self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): Is the input data
            Y (numpy.ndarray): Is the correct labels for the input data
            A (numpu.ndarray): Is the activated output of the neuron with
            shape (1, m) where m is the number of examples
            alpha (float): Is the learning rate.

        """
        m = len(Y[0])
        grad = np.matmul(X, (A - Y).T) / m
        self.__W -= alpha * grad.T
        self.__b -= alpha * np.average(A - Y)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron

        Args:
            X (numpy.ndarray): Is the input data with shape (nx, m) where
            nx is the number of features and m the number of exapmles.
            Y (numpy.ndarray): Is the correct labels for the input data with
            shape (1, m) where m is the number of examples
            iterations (int): Is the number of iterations to train over.
            alpha (float): Is the learning rate.
            verbose (boolean): Ask If wether or not to print information about
            the training.
            graph (boolean): Is defines wether or not graph information about
            the training.
            step (int): Is the number of steps it should print or graph
            information about training.

        Raises:
            TypeError: If iterations is not an integer.
            ValueError: If iterations is not positive number.
            TypeError: If alpha is not a float.
            ValueError: if alpha is not positive number
            TypeError: If step is not an integer.
            ValueError: If step is not positive or superior then iterations.

        Returns:
            float: The evaluation of the training data after iterations of
            training have occured

        """
        costs = np.array(())
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step not in range(0, iterations + 1):
                raise ValueError('step must be positive and <= iterations')
        for iteration in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            if graph:
                costs = np.append(costs, cost)
            if verbose and step in range(0, iterations + 1):
                print("Cost after {} iterations: {}".format(iteration, cost))
            self.gradient_descent(X, Y, A, alpha)
        if graph:
            plt.plot(costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
        return self.evaluate(X, Y)
