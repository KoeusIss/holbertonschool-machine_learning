#!/usr/bin/env python3
"""Attention module"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer

    Arguments:
        max_seq_len {int} -- The maximum sequence length
        dm {int} -- Is the model length

    Returns:
        np.ndarray -- Containing the positional encoding matrix
    """
    position = np.arange(max_seq_len)
    min_freq = 1e-4
    frequences = np.power(min_freq, 2 * (np.arange(dm) // 2) / dm)
    PE = position.reshape(-1, 1) * frequences.reshape(1, -1)
    PE[:, ::2] = np.cos(PE[:, ::2])
    PE[:, 1::2] = np.sin(PE[:, 1::2])
    return PE
