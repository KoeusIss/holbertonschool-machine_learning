#!/usr/bin/env python3
"""Inception CNN module"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block.

    Args:
        A_prev: Is the ouput of the previous layer.
        filters (list|tuple): Is containing filters dimension of shape
            (F1, F3R, F3, F5R, F5, FPP) where
            F1: number of filters for 1X1 convolution in first branch
            F3R: number of filters for 1X1 convolution in second branch
            F3: number of filters for 3X3 convolution in second branch
            F5R: number of filters for 1X1 convolution in third branch
            F5: number of filters for 5X5 convolution in third branch
            FPP: number of filters for MaxPooling convolution in fourth branch

    Returns:
        keras.model: Concatenated output

    """
    F1, F3R, F3, F5R, F5, FPP = filters

    C1X1a = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    C1X1b = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    C3X3b = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        activation='relu',
        padding='same',
    )(C1X1b)
    C1X1c = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(A_prev)
    C5X5c = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        activation='relu',
        padding='same',
    )(C1X1c)
    MPd = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding='same'
    )(A_prev)
    C1X1d = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        activation='relu',
        padding='same',
    )(MPd)
    return K.layers.Concatenate()([C1X1a, C3X3b, C5X5c, C1X1d])
