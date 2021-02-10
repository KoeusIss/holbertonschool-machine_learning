#!/usr/bin/env python3
"""DenseNet module"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture

    Args:
        growth_rate (float): Is the growth rate.
        compression (float): Is the compression factor.

    Returns:
        keras.model

    """
    X = K.Input(shape=[224, 224, 3])
    X_BN = K.layers.BatchNormalization()(X)
    X_relu = K.layers.Activation('relu')(X_BN)

    conv_layer = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer='he_normal'
    )(X_relu)
    conv_MP = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_layer)
    nb_filters = conv_MP.shape[-1].value
    dense_blk_1, nb_filters_1 = dense_block(
        conv_MP,
        nb_filters,
        growth_rate,
        6
    )
    transition_layer_2, nb_filters_2 = transition_layer(
        dense_blk_1,
        nb_filters_1,
        compression
    )
    dense_blk_3, nb_filters_3 = dense_block(
        transition_layer_2,
        nb_filters_2,
        growth_rate,
        12
    )
    transition_layer_4, nb_filters_4 = transition_layer(
        dense_blk_3,
        nb_filters_3,
        compression
    )
    dense_blk_5, nb_filters_5 = dense_block(
        transition_layer_4,
        nb_filters_4,
        growth_rate,
        24
    )
    transition_layer_6, nb_filters_6 = transition_layer(
        dense_blk_5,
        nb_filters_5,
        compression
    )
    dense_blk_7, nb_filters_7 = dense_block(
        transition_layer_6,
        nb_filters_6,
        growth_rate,
        16
    )
    conv_AP = K.layers.AveragePooling2D(
        pool_size=7,
    )(dense_blk_7)
    fully_connected = K.layers.Dense(
        units=1000,
        activation='softmax'
    )(conv_AP)

    return K.models.Model(inputs=X, outputs=fully_connected)
