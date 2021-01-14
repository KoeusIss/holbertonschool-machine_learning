#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network using
    tensorflow

    Args:
        prev (tf.Tensor): Is the activated output of the previous layer
        n (int): Is the number of nodes in the layer be created.
        acitvation (tf.nn.activation): Is the activation function that should
            be used on the output of the layer

    Returns:
        tf.Tensor: Is the activated output of the layer

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=init)
    layered = layer(prev)
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]), name="gamma", trainable=True
    )
    beta = tf.Variable(
        tf.constant(0.0, shape=[n]), name="beta", trainable=True
    )
    mean, variance = tf.nn.moments(layered, axes=[0])
    epsilon = 1e-8
    normed = tf.nn.batch_normalization(
        layered, mean, variance, beta, gamma, epsilon
    )
    return activation(normed)
