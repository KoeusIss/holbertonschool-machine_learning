#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def crop_image(image, size):
    """Crops randomly an image to the given size.

    Arguments:
        image {tf.tensor} -- The Given image to crop.
        size {tuple(int)} -- Is the needed size.

    Returns:
        tf.Tensor -- The Croped image.
    """
    return tf.image.random_crop(image, size=size)
