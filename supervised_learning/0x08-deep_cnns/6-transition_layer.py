#!/usr/bin/env python3
"""DenseNet module"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer

    Args:
        X (keras.model): Is the output of the previous layer
        nb_filters (int): Is the number of filters on the previous layer
        compression (float): Is the compression factor for the transition
            layer.

    Returns:
        keras.model: The output of the trasition layer

    """
    X_BN = K.layers.BatchNormalization()(X)
    X_relu = K.layers.Activation('relu')(X_BN)

    conv1 = K.layers.Conv2D(
        filters=int(nb_filters * compression),
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal'
    )(X_relu)
    conv_AP = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2
    )(conv1)

    return conv_AP, nb_filters
