#!/usr/bin/env python3
"""Convolutional Neural Network module"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the `LeNet-5` architecture using Keras
    
    Args:
        X (keras.input): Containing the input images for the network with shape
            (m, 28, 28, 1) where m is the number of images.
    
    Returns:
        keras.model: A keras model compiled to use `Adam` optimization and
        accuracy mertics.
    
    """
