#!/usr/bin/env python3
"""RNN module"""
import numpy as np


class BidirectionalCell:
    """BidirectionalCell class"""
    def __init__(self, i, h, o):
        """Initializer

        Arguments:
            i {int} -- Is dimensionality of the data
            h {int} -- Is dimensionality of hidden state
            o {int} -- Is dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        Arguments:
            h_prev {np.ndarrat} -- Contains the previous hidden state
            x_t {np.ndarray} -- Contains the data input of the cell

        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, the output of
            the cell
        """
        stacked = np.hstack((h_prev, x_t))
        h_next = np.tanh(stacked @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Performs backward propagation for one time step

        Arguments:
            h_next {np.ndarrat} -- Contains the next hidden state
            x_t {np.ndarray} -- Contains the data input of the cell

        Returns:
            tuple(np.ndarray) -- Contains the prev hidden state, the output of
            the cell
        """
        stacked = np.hstack((h_next, x_t))
        h_prev = np.tanh(stacked @ self.Whb + self.bhb)
        return h_prev
