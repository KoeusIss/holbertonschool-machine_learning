#!/usr/bin/env python3
"""Convolutional Neural Network module"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the `LeNet-5` architecture using Keras

    Args:
        X (keras.input): Containing the input images for the network with shape
            (m, 28, 28, 1) where m is the number of images.

    Returns:
        keras.model: A keras model compiled to use `Adam` optimization and
        accuracy mertics.

    """
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(X)
    M1 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C1)
    C2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        strides=(2, 2),
        activation='relu',
        kernel_initializer='he_normal'
    )(M1)
    M2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(C2)
    CF = K.layers.Flatten()(C2)
    F1 = K.layers.Dense(
        units=84,
        kernel_initializer="he_normal",
        activation='relu'
    )(CF)
    F2 = K.layers.Dense(
        units=120,
        kernel_initializer="he_normal",
        activation='relu'
    )(F1)
    F3 = K.layers.Dense(
        units=10,
        kernel_initializer="he_normal",
    )(F2)
    model = K.Model(inputs=X, outputs=F3)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=K.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model
