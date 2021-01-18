#!/usr/bin/env python3
"""Error Analysis module"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in confusion matrix

    Args:
        confusion (numpy.ndarray): Is containing a confusion matrix of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: Is containing the sensitivity of each classes of shape
            (classes,)

    """
    classes = confusion.shape[0]
    return np.sum(np.eye(classes) * confusion, axis=1)\
        / np.sum(confusion, axis=1)
