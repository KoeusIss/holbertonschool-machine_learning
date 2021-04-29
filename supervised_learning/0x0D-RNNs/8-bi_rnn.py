#!/usr/bin/env python3
"""RNN module"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs a Bidirectional RNN

    Arguments:
        bi_cell {BidirectionalCell} -- Instance of bidirectional cell
        X {np.ndarray} -- Contains the input data
        h_0 {np.ndarray} -- Contain the initial hidden state
        h_t {np.ndarray} -- Contain the initial hidden state

    Returns:
        tuple -- Contains all the hidden states, and the output of the net.
    """
    T, m, i = X.shape
    _, s_0 = h_0.shape
    _, s_t = h_t.shape

    Hf = np.zeros((T, m, s_0))
    Hb = np.zeros((T, m, s_t))

    for t in range(0, T):
        Hf[t] = bi_cell.forward(h_0, X[t])
        h_0 = Hf[t]

    for t in range(0, T)[::-1]:
        Hb[t] = bi_cell.backward(h_t, X[t])
        h_t = Hb[t]

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)
    return H, Y
