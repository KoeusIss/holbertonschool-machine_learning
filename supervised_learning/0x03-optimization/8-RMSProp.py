#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using RMSprop optimization algorithm

    Args:
        loss (tf.Tensor): Is the loss of the network.
        alpha (float): Is the learning rate
        beta2 (float): Is the RMSProp weight
        epsilon (float): Is the small number to prevent division by zero

    Returns:
        tf.operation: The RMSProp operation.

    """
    rms_prop = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return rms_prop.minimize(loss)
