#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using Adam algorithm

    Args:
        loss (tf.Tensor): Is the loss of the network
        alpha (float): Is the learning rate
        beta1 (float): Is the weight for the first moment
        beta2 (float): Is the weight for the second moment
        epsilon (float): Is the small number to avoid division by zero

    Returns:
        tf.operation: Adam optimization operation

    """
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)
