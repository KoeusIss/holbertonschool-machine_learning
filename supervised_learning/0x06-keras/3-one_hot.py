#!/usr/bin/env python3
"""Keras module"""
import numpy as np


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix

    Args:
        labels (numpy.ndarray): Is containing the input labels.
        classes (int): number of classes

    Returns:
        numpy.ndarray: The one-hot matrix.

    """
    m = labels.shape[0]
    if classes is None:
        classes = len(np.unique(labels, axis=0))
    return np.eye(classes)[labels]
