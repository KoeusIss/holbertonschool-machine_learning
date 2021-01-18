#!/usr/bin/env python3
"""Error Analysis module"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the f1_score for each class in a confusion matrix

    Args:
        confusion (numpy.ndarray): Is a confusion matrix of shape
            (classes, classes) where the row indices is the correct labels
            and the column indices represent the predicted label

    Returns:
        numpy.ndarray: Containing the f1_score of each class of shape
            (classes,)

    """
    return 2 / (precision(confusion)**(-1) + sensitivity(confusion)**(-1))
