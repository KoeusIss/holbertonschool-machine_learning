#!/usr/bin/env python3
"""ResNet module"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an Identity block

    Args:
        A_prev: The output of previous layer
        filters (list|tuple): Contains the number of filters of each
            convolution inside the block

    Returns:
        keras.model: The ouput of the identity block

    """
    F11, F3, F13 = filters

    C1x1_1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        kernel_initializer='he_normal',
    )(A_prev)
    C1x1_1_BN = K.layers.BatchNormalization()(C1x1_1)
    C1x1_1_relu = K.layers.Activation('relu')(C1x1_1_BN)

    C3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same'
    )(C1x1_1_relu)
    C3x3_BN = K.layers.BatchNormalization()(C3x3)
    C3x3_relu = K.layers.Activation('relu')(C3x3_BN)

    C1x1_2 = K.layers.Conv2D(
        filters=F13,
        kernel_size=1,
        kernel_initializer='he_normal',
    )(C3x3_relu)
    C1x1_2_BN = K.layers.BatchNormalization()(C1x1_2)

    layer_addition = K.layers.Add()([A_prev, C1x1_2_BN])
    return K.layers.Activation('relu')(layer_addition)
