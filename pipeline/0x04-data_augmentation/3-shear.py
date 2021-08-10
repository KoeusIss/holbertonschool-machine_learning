#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def shear_image(image, intensity):
    """Shears an image randomly

    Arguments:
        image {tf.Tensor} -- Contains the given image
        intensity {int} -- Contains the max intensity shear

    Returns:
        tf.Tensor -- The shared images
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity)
