#!/usr/bin/env python3
"""Inception CNN module"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network

    Returns:
        keras.module

    """
    X = K.Input(shape=(224, 224, 3))

    C7x7_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu'
    )(X)
    MaxP_2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(C7x7_1)
    C1x1_3 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(MaxP_2)
    C3x3_4 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    )(C1x1_3)
    MaxP_5 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(C3x3_4)
    In3a_6 = inception_block(
        MaxP_5,
        (64, 96, 128, 16, 32, 32)
    )
    In3b_7 = inception_block(
        In3a_6,
        (128, 128, 192, 32, 96, 64)
    )
    MaxP_8 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(In3b_7)
    In4a_9 = inception_block(
        MaxP_8,
        (192, 96, 208, 16, 48, 64)
    )
    In4b_10 = inception_block(
        In4a_9,
        (160, 112, 224, 24, 64, 64)
    )
    In4c_11 = inception_block(
        In4b_10,
        (128, 128, 256, 24, 64, 64)
    )
    In4d_12 = inception_block(
        In4c_11,
        (112, 144, 288, 32, 64, 64)
    )
    In4e_13 = inception_block(
        In4d_12,
        (256, 160, 320, 32, 128, 128)
    )
    MaxP_14 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(In4e_13)
    In5a_15 = inception_block(
        MaxP_14,
        (256, 160, 320, 32, 128, 128)
    )
    In5b_16 = inception_block(
        In5a_15,
        (384, 192, 384, 48, 128, 128)
    )
    AvgP_17 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=1,
        padding='same'
    )(In5b_16)
    Drop_18 = K.layers.Dropout(
        rate=0.4,
    )(AvgP_17)
    FulC_19 = K.layers.Dense(
        units=1000,
        activation='softmax'
    )(Drop_18)

    return K.models.Model(inputs=X, outputs=FulC_19)
