#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates the learning rate decay operation in tensorflow using inverse
    time decay

    Args:
        alpha (float): Is the original learning rate
        decay_rate (float): Is the weight used to determine the rate at which
            alpha will decay.
        global_step (int): Is the number of passes of gradient descent that
            have elapsed.
        decay_step (int): Is the number of passes of gradient descent that
            occur before alpha is decayed further.

    Returns:
        tf.operation: The learning rate decay operation

    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate
    )
