#!/usr/bin/env python3
"""One-Hot decode module"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts one hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): Is a one hot encoded

    """
    if type(one_hot) != np.ndarray:
        return None
    try:
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
