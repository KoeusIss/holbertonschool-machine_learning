#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def crop_image(image, size):
    return tf.image.resize(image, size=size)
