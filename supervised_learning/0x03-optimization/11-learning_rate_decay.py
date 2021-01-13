#!/usr/bin/env python3
"""Optimization module"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy

    Args:
        alpha (float): Is the learning rate
        decay_rate (float): Is the weight used to determine the rate at which
            alpha will decay
        global_stop (int): Is the number of passes of gradient descent that
            have elapsed.
        decay_stop (int): Is the number of passes of gradient descent that
            should occur before alpha is decayed further

    Returns
        float: the updated alpha

    """
    return alpha / (1 + decay_rate * global_step // decay_step)
