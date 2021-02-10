#!/usr/bin/env python3
"""DenseNet module"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds dense block

    Args:
        X (keras.Input): the ouput of the previous layer
        nb_filters (int): Is the number of filters in X
        growth_rate (int): Is the growth rate of the dense block
        layers (int): Is the number of layers in the dense block

    Returns:
        keras.model: The concatenated layers

    """
    inputs = X

    for layer in range(layers):
        inputs_BN = K.layers.BatchNormalization()(inputs)
        inputs_relu = K.layers.Activation('relu')(inputs_BN)
        conv1 = K.layers.Conv2D(
            filters=128,
            kernel_size=1,
            padding='same',
            kernel_initializer='he_normal'
        )(inputs_relu)
        conv1_BN = K.layers.BatchNormalization()(conv1)
        conv1_relu = K.layers.Activation('relu')(conv1_BN)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal'
        )(conv1_relu)

        inputs = K.layers.Concatenate()([inputs, conv2])
        nb_filters += growth_rate

    return inputs, nb_filters
