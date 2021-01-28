#!/usr/bin/env python3
"""Convolution and pooling podule"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images

    Args:
        images (numpy.ndarray): Containing multiple gray scale images of shape
            (m, h, w) where m is the number of images, h is the height in
            pixels of the images and w is the width in pixels of images.
        kernel (numpy.ndarray): Containing the kernel for the convolution of
            shape (kh, kw) where kh is the height of the kernel and kw is the
            widht of the kernel.

    Returns:
        numpy.ndarray: Containing the convolved image.

    """
    m, input_h, input_w = images.shape
    kh, kw = kernel.shape

    output_h = int(np.floor(input_h - kh + 1))
    output_w = int(np.floor(input_w - kw + 1))
    output = np.zeros((m, output_h, output_w))
    for w in range(output_w):
        for h in range(output_h):
            output[:, h, w] = np.sum(
                kernel * images[:, h:h + kh, w:w + kw],
                axis=(1, 2)
            )
    return output
