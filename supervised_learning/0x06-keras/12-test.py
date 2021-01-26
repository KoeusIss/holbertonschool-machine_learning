#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network

    Args:
        network (keras.model): Is the network model
        data (): Is the input data.
        labels (): Is the correct label of data
        verbose (boolean): Determines whether the output should be printed.

    Returns:
        float/float: Loss and accuracy of the model

    """
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
