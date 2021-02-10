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
    conv2_1 = projection_block(
        conv2_MP,
        [64, 64, 256],
        1
    )
    conv2_2 = identity_block(
        conv2_1,
        [64, 64, 256],
    )
    conv2_3 = identity_block(
        conv2_2,
        [64, 64, 256],
    )
    conv3_1 = projection_block(
        conv2_3,
        [128, 128, 512],
    )
    conv3_2 = identity_block(
        conv3_1,
        [128, 128, 512],
    )
    conv3_3 = identity_block(
        conv3_2,
        [128, 128, 512],
    )
    conv3_4 = identity_block(
        conv3_3,
        [128, 128, 512],
    )
    conv4_1 = projection_block(
        conv3_4,
        [256, 256, 1024],
    )
    conv4_2 = identity_block(
        conv4_1,
        [256, 256, 1024],
    )
    conv4_3 = identity_block(
        conv4_2,
        [256, 256, 1024],
    )
    conv4_4 = identity_block(
        conv4_3,
        [256, 256, 1024],
    )
    conv4_5 = identity_block(
        conv4_4,
        [256, 256, 1024],
    )
    conv4_6 = identity_block(
        conv4_5,
        [256, 256, 1024],
    )
    conv5_1 = projection_block(
        conv4_6,
        [512, 512, 2048],
    )
    conv5_2 = identity_block(
        conv5_1,
        [512, 512, 2048],
    )
    conv5_3 = identity_block(
        conv5_2,
        [512, 512, 2048],
    )
    conv_AP = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same'
    )(conv5_3)
    fully_connected = K.layers.Dense(
        units=1000,
        activation='softmax',
    )(conv_AP)

    return K.models.Model(inputs=X, outputs=fully_connected)
