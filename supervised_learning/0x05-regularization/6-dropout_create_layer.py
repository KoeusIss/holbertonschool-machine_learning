#!/usr/bin/env python3
"""Regularization module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using Dropout

    Args:
        prev (tf.Tensor): Is a tensor containing the ouput of the previous
            layer.
        n (int): Is the number nodes units in the new layer.
        activation: Is the activation function that should be used on the layer
        keep_prob (float): Is the probability that a node will be kept.

    Returns:
        tf.layer: The output of the new layer.

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )
    logits = layer(prev)
    dropout = tf.layers.Dropout(rate=(1 - keep_prob))
    return dropout(logits)
