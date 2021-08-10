#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Adjusts image's Brightness

    Arguments:
        image {tf.Tensor} -- Contains the given image.
        max_delta {float} -- Is the max delta brightness.

    Returns:
        tf.tensor -- Adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
