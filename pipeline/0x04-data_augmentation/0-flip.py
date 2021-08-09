#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def flip_image(image):
    """Flips an image horizontally

    Arguments:
        image {tf.tensor} -- Contains the given image image

    Returns:
        tf.tensor -- The flipped image
    """
    return tf.image.flip_left_right(image)
