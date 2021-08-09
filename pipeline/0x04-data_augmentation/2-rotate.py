#!/usr/bin/env python3
"""Data augmentation module"""
import tensorflow as tf


def rotate_image(image):
    return tf.image.rot90(image, k=1)
