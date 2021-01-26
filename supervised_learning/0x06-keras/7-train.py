#!/usr/bin/env python3
"""Keras module"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using the mini-batch gradient descent and analyze
    validation data.

    Args:
        network (keras.model): Is the model to train.
        data (numpy.ndarray): Is containing the input data of shapr (m, nx)
            where m is the number of input data and nx is the number of
            features.
        labesl (numpy.ndarray): Is containing the one-hot labels of data.
        batch_size (int): Is the size of batch used for mini-batch gradient
            descent.
        epochs (int): Is the number of passes through the data for MBGD.
        validation_data (numpy.ndarray): Is the data to validate the model with
        early_stoppint (boolean): determines whether early stopping should be
            used or not.
        patience (int): Is the patience used for early stopping.
        verbose (boolean): Is determines if output should be printed during
            training.
        shuffle (boolean): Is determines whether to shuffle the batches every
            epoch, It is set to False for reproducibility.
        learning_rate_decay (boolean): indicates whether learning rate should
            be used or not.
        alpha (float): Is the initial learning rate.
        decay_rate (float): Is the decay rate.

    Returns:
        Objects: History objects generated after training.

    """
    callback = early_stopping and K.callbacks.EarlyStopping(
        patience=patience
    )
    itd = K.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=epochs,
        decay_rate=decay_rate,
        staircase=True
    )
    history = network.fit(
        x=data,
        y=labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=None,
        shuffle=shuffle,
        callbacks=[callback, itd]
    )
    return history