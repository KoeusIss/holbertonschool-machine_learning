#!/usr/bin/env python3
"""Convoltions and pooling module"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels

    Args:
        images (numpy.ndarray): Containing multiple gray scale images of shape
            (m, h, w) where m is the number of images, h is the height in
            pixels of the images and w is the width in pixels of images.
        kernel (numpy.ndarray): Containing the kernel for the convolution of
            shape (kh, kw) where kh is the height of the kernel and kw is the
            widht of the kernel.
        padding (tuple): containing (ph, pw) where ph is the padding along the
            height of the image, and the pw is the padding along the width.

    Returns:
        numpy.ndarray: Containing the convolved image.

    """
