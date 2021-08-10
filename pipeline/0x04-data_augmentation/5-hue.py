#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def change_hue(image, delta):
    """Adjusts the hue of an image.

    Arguments:
        image {tf.Tensor} -- Contains the given image
        delta {int} -- is the hue delta.

    Returns:
        tf.tensor -- The adjusted image
    """
    return tf.image.adjust_hue(image, delta)
