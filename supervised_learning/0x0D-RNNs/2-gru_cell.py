#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class GRUCell:
    """GRUCell class
    """
    def __init__(self, i, h, o):
        """Initializer

        Arguments:
            i {int} -- Is the dimensionality of the data
            h {int} -- Is the dimensionality of the hidden state
            o {int} -- Is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(z):
        """Computes the sigmoid

        Arguments:
            z {np.ndarray} -- Is contain the input

        Returns:
            np.ndarray -- Contains the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        """Calculates the softmax

        Arguments:
            z {np.ndarray} -- Contains array input

        Returns:
            np.ndarray -- Softmaxed numpy arrat
        """
        ex = np.exp(z)
        return ex / np.sum(ex, 1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perfoms forward propagation for one time step

        Arguments:
            h_prev {np.ndarray} -- Contains the previous hidden state
            x_t {np.ndarray} -- Contains the data

        Returns:
            tuple -- Contains the next hidden state, the ouput of the cell
        """
        stacked = np.hstack((h_prev, x_t))
        z = self.sigmoid(stacked @ self.Wz + self.bz)
        r = self.sigmoid(stacked @ self.Wr + self.br)
        stacked = np.hstack((r * h_prev, x_t))
        h_tild = np.tanh(stacked @ self.Wh + self.bh)
        h_next = (np.ones_like(z) - z) * h_prev + z * h_tild
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, y
