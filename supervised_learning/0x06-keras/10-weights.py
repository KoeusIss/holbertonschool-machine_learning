#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves model's weights

    Args:
        network (keras.model): Is the model
        filename (str): Is the file path where should weights be saved
        save_format (str): The saving format

    Returns:
        None

    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Loads a model's weights

    Args:
        network (keras.model): The model which the weights should be loaded
        filename (str): Is the path of the file that should be loaded from.

    Returns:
        None

    """
    network.load_weights(filename)
    return None
