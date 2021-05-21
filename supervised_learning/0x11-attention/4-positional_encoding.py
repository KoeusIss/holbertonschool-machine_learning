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
    def get_angles(pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dm))
        return pos * angle_rates

    position = np.arange(max_seq_len)
    PE = get_angles(position[:, np.newaxis], np.arange(dm)[np.newaxis, :])
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    return PE
