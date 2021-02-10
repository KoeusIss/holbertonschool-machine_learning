#!/usr/bin/env python3
"""ResNet module"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block

    Args:
        A_prev: The output of the previous layer
        filters (list|tuple): Containing the number of filters in each layer
        s (int): Is the stride of the first layer

    Returns:
        keras.model: The activated output

    """
    F11, F3, F12 = filters

    C1x1_1a = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        kernel_initializer='he_normal',
        padding='same'
    )(A_prev)
    C1x1_1a_BN = K.layers.BatchNormalization()(C1x1_1a)
    C1x1_1a_relu = K.layers.Activation('relu')(C1x1_1a_BN)

    C3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer='he_normal'
    )(C1x1_1a_relu)
    C3x3_BN = K.layers.BatchNormalization()(C3x3)
    C3x3_relu = K.layers.Activation('relu')(C3x3_BN)

    C1x1_2a = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        kernel_initializer='he_normal',
        padding='same'
    )(C3x3_relu)
    C1x1_2a_BN = K.layers.BatchNormalization()(C1x1_2a)

    C1x1_1b = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        kernel_initializer='he_normal',
        padding='same'
    )(A_prev)
    C1x1_1b_BN = K.layers.BatchNormalization()(C1x1_1b)

    path_addition = K.layers.Add()([C1x1_2a_BN, C1x1_1b_BN])
    return K.layers.Activation('relu')(path_addition)
