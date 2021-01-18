#!/usr/bin/env python3
"""Error Analysis module"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates the confusion matrix

    Args:
        labels (numpy.ndarray): Is containing the correct labels for each data
            point of shape (m, classes) where m is the number of data points
            and classes is the number of classes.
        logits (numpy.ndarray): Is containing the predicted labels of shape
            (m, classes) where m is the number of data points and classes is
            the number of calsses.

    Returns:
        numpy.ndarray: A confusion matrix of shape (classes, classes) where
            classes is the number of classes.

    """
    return np.matmul(labels.T, logits)
