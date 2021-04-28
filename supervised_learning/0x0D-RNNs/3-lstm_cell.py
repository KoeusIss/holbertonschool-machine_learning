#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class LSTMCell:
    """LSTMCell class
    """
    def __init__(self, i, h, o):
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation on LSTM cell

        Arguments:
            h_prev {np.ndarray} -- Contains the previous hidden states
            c_prev {np.ndarray} -- Contains the previous cell state
            x_t {np.ndarray} -- Contains the data

        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, next cell
            state and the cell output
        """
        stacked = np.hstack((h_prev, x_t))
        f = self.sigmoid(stacked @ self.Wf + self.bf)
        u = self.sigmoid(stacked @ self.Wu + self.bu)
        c_tild = np.tanh(stacked @ self.Wc + self.bc)
        c_next = f * c_prev + u * c_tild
        o = self.sigmoid(stacked @ self.Wo + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(h_next @ self.Wy + self.by)
        return h_next, c_next, y
