#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix

    Args:
        labels (numpy.ndarray): Is containing the input labels.
        classes (int): number of classes

    Returns:
        numpy.ndarray: The one-hot matrix.

    """
    return K.utils.to_categorical(labels, classes)
