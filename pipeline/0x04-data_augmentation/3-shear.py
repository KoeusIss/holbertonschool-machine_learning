#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def shear_image(image, intensity):
    return tf.keras.preprocessing.image.random_shear(image, intensity)
