#!/usr/bin/env python3
"""Keras module"""
import tensorflow as tf
import tensorflow.keras as keras


def train_model(network, data, labels,
                batch_size, epochs, verbose=True, shuffle=False):
    """Trains a model using the mini-batch gradient descent

    Args:
        network (keras.model): Is the model to train.
        data (numpy.ndarray): Is containing the input data of shapr (m, nx)
            where m is the number of input data and nx is the number of
            features.
        labesl (numpy.ndarray): Is containing the one-hot labels of data.
        batch_size (int): Is the size of batch used for mini-batch gradient
            descent.
        epochs (int): Is the number of passes through the data for MBGD.
        verbose (boolean): Is determines if output should be printed during
            training.
        shuffle (boolean): Is determines whether to shuffle the batches every
            epoch, It is set to False for reproducibility.

    Returns:
        Objects: History objects generated after training.

    """
    history = network.fit(
        x=data,
        y=labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
