#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for keras model with categorical crossentropy
    loss and accuracy metrics.

    Args:
        network (keras.model): Is the model to optimize
        alpha (float): Is the learning rate.
        beta1 (float): Is the first Adam optimization parameter
        beta2 (float): Is the second Adam optimization parameter

    Returns:
        None

    """
    adam = K.optimizers.Adam(
        lr=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=['accuracy']
    )
    return None
