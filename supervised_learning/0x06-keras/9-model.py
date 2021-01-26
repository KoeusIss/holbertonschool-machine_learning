#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model

    Args:
        network (keras.model): Is the model to save
        filename (str): Is the path of the file where the model should be save

    """
    network.save(filename)


def load_model(filename):
    """Loads an entire model
    
    Args:
        filename (str): Is the path of the file that the model should be loaded
    
    Returns:
        keras.model: the loaded model
    """
    return K.models.load_model(filename)
