#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediciton a neural network

    Args:
        network (keras.model): Is the network model
        data (): Is the input data.
        verbose (boolean): Determines whether the output should be printed.

    Returns:
        list: the prediction list

    """
    return network.predict(
        x=data,
        verbose=verbose
    )
