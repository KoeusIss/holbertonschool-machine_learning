#!/usr/bin/env python3
"""Regularization module"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization

    Args:
        prev (tf.Tensor): Is containing the output of the previous layer.
        n (int): Is the number of nodes the new layer should contain
        activation: Is the activation function that should be used on the layer
        lambtha (float): Is the l2 regularization parameter

    Returns:
        tf.layer: The output of the new layer

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=reg,
        name="layer"
    )
    return layer(prev)
