#!/usr/bin/env python3
"""Keras module"""
import tensorflow as tf
import tensorflow.keras as keras


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
    inputs = keras.Input(shape=(nx,))
    regularizer = keras.regularizers.l2(lambtha)
    outputs = inputs
    for layer, activation in zip(layers, activations):
        outputs = keras.layers.Dense(
            layer,
            activation,
            kernel_regularizer=regularizer
        )(outputs)
        if layers.index(layer) < len(layers) - 1:
            dropout = keras.layers.Dropout((1 - keep_prob))
            outputs = dropout(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)
