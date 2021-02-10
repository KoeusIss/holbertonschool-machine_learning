#!/usr/bin/env python3
"""ResNet module"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture

    Returns:
        keras.model

    """
    X = K.Input(shape=[224, 224, 3])

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        kernel_initializer='he_normal',
        strides=2,
        padding='same'
    )(X)
    conv1_BN = K.layers.BatchNormalization()(conv1)
    conv1_relu = K.layers.Activation('relu')(conv1_BN)

    conv2_MP = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(conv1_relu)
    conv2_1 = identity_block(
        conv2_MP,
        [64, 64, 256]
    )
    conv2_2 = identity_block(
        conv2_1,
        [64, 64, 256]
    )
    conv2_3 = identity_block(
        conv2_2,
        [64, 64, 256]
    )
    
    output = conv2_3

    return K.models.Model(inputs=X, outputs=output)
