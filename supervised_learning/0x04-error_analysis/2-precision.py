#!/usr/bin/env python3
"""Error Analysis module"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix

    Args:
        confusion (numpy.ndarray): Is a confusion matrix of shape
            (classes, classes) where the row indices is the correct labels
            and the column indices represent the predicted label

    Returns:
        numpy.ndarray: Containing the precision of each class of shape
            (classes,)

    """
    classes = confusion.shape[0]
    return np.sum(np.eye(classes) * confusion, axis=1)\
        / np.sum(confusion, axis=0)
