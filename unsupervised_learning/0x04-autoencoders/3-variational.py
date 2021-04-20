#!/usr/bin/env python3
"""Autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder

    Arguments:
        input_dims {int} -- Is the input size
        hidden_layers {list} -- Contain the number of nodes for each layer
        latent_dims {int} -- The size of latent space

    Returns:
        tupe -- encoder, decoder, and autoencoder model
    """
    inputs = keras.Input(shape=(input_dims,))
    h = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    for dim in hidden_layers[1:]:
        h = keras.layers.Dense(dim, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0.,
            stddev=0.1
        )
        return z_mean + keras.backend.exp(z_log_sigma) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[-1], activation='relu')(latent_inputs)
    for dim in hidden_layers[-2::-1]:
        x = keras.layers.Dense(dim, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)
    auto = keras.Model(inputs, decoder(encoder(inputs)[2]))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto