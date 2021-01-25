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
    model = keras.Sequential()
    regularizer = keras.regularizers.l2(lambtha)
    for layer, activation in zip(layers, activations):
        if layers.index(layer) == 0:
            model.add(keras.layers.Dense(
                layer,
                activation,
                input_shape=(nx,),
                kernel_regularizer=regularizer
            ))
        else:
            model.add(keras.layers.Dense(
                layer,
                activation,
                kernel_regularizer=regularizer
            ))
        if layers.index(layer) < len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))
    return model
