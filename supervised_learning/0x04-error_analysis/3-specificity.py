#!/usr/bin/env python3
"""Error Analysis module"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix

    Args:
        confusion (numpy.ndarray): Is a confusion matrix of shape
            (classes, classes) where the row indices is the correct labels
            and the column indices represent the predicted label

    Returns:
        numpy.ndarray: Containing the specificity of each class of shape
            (classes,)

    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return TN / (TN + FP)
