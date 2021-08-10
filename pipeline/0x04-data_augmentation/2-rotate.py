#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image 90 degres anticlockwise.

    Arguments:
        image {tf.Tensor} -- Contains the given image

    Returns:
        tf.Tensor -- The rotated image
    """
    return tf.image.rot90(image, k=1)
