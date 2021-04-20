#!/usr/bin/env python3
"""Autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates an autoencoder

    Arguments:
        input_dims {int} -- Is the dimensions of the model input
        hidden_layers {int} -- Is a list containing the number of nodes of
        each hidden layer in the encoder.
        latent_dims {int} -- Is containing the dimensions of the latent space
        representation

    Returns:
        tuple(keras.model) -- encoder, decoder and autoencoder models
    """
    regularizer = keras.regularizers.l1(lambtha)

    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu',
        activity_regularizer=regularizer
    )(inputs)
    for dim in hidden_layers[1:]:
        encoded = keras.layers.Dense(
            dim,
            activation='relu',
            activity_regularizer=regularizer
        )(encoded)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizer
    )(encoded)

    encoder = keras.Model(inputs, latent)

    encoded_input = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1],
        activation='relu',
        activity_regularizer=regularizer
    )(encoded_input)
    for dim in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(
            dim,
            activation='relu',
            activity_regularizer=regularizer
        )(decoded)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(encoded_input, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto