#!/usr/bin/env python3
"""RNN module"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN

    Arguments:
        rnn_cell {RNNCell} -- RNNCell instance
        X {np.ndarray} -- Contains the data to be used
        h_0 {np.ndarray} -- Contains the initial hidden state

    Returns:
        tuple(np.ndarrat) -- Contains all the hidden states, and the
        outputs
    """
    T, m, i = X.shape
    h, o = rnn_cell.Wy.shape

    H = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, o))
    for t in range(1, T + 1):
        H[t], Y[t - 1] = rnn_cell.forward(H[t - 1], X[t - 1])
    return H, Y
