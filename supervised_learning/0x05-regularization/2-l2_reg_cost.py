#!/usr/bin/env python3
"""Regularization module"""
import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization using
    tensorflow.

    Args:
        cost (tf.Tensor): Is containing the cost of the network without L2
            regularization

    Returns:
        tf.Tensor: The regularized cost

    """
    return cost + tf.losses.get_regularization_loss()
