#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with a keras library

    Args:
        nx (int): Is the number of the input features of the network.
        layers (list): Is containing the number of nodes in each layers.
        activations (list): Is containing functions used for each layer.
        lambtha (float): Is the L2 regularization parameter.
        keep_prob (floa): Is the probability that a node will be kept.

    Returns:
        keras.model: Returns the keras model

    """
    inputs = K.Input(shape=(nx,))
    outputs = inputs
    regularizer = K.regularizers.l2(lambtha)
    for layer, activation in zip(layers, activations):
        outputs = K.layers.Dense(
            layer,
            activation,
            kernel_regularizer=regularizer
        )(outputs)
        if layers.index(layer) < len(layers) - 1:
            dropout = K.layers.Dropout((1 - keep_prob))
            outputs = dropout(outputs)
    return K.Model(inputs=inputs, outputs=outputs)
