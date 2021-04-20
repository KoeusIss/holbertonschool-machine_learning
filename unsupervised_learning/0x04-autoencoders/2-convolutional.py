#!/usr/bin/env python3
"""Autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Arguments:
        input_dims {tuple} -- Contains the input dimension
        filters {list} -- Contain the layer number of filters
        latent_dims {tuple} -- Contain the dimension of latent space

    Returns:
        tuple -- encoder, decoder and autoencoder model
    """
    inputs = keras.Input(shape=input_dims)

    encoded = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(inputs)
    encoded = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )(encoded)
    for dim in filters[1:]:
        encoded = keras.layers.Conv2D(
            filters=dim,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(encoded)
        encoded = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(encoded)

    encoder = keras.Model(inputs, encoded)
    encoded_input = keras.Input(shape=latent_dims)

    decoded = keras.layers.Conv2D(
        filters=filters[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(encoded_input)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    for dim in filters[-2:0:-1]:
        decoded = keras.layers.Conv2D(
            filters=dim,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid',
    )(decoded)

    decoder = keras.Model(encoded_input, outputs)
    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
