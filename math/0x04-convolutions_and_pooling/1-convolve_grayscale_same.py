#!/usr/bin/env python3
"""Convolution and pooling podule"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images

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

    output_h = input_h
    output_w = input_w

    padding_h = kh // 2
    padding_w = kw // 2

    padded_images = np.pad(
        array=images,
        pad_width=((0,), (padding_h,), (padding_w,)),
        mode="constant",
        constant_values=0
    )

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                kernel * padded_images[:, i:i + kh, j:j + kw],
                axis=(1, 2)
            )
    return output
