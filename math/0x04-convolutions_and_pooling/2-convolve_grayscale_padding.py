#!/usr/bin/env python3
"""Convolutions and pooling module"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

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
    m, input_h, input_w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = input_h - kh + 2 * ph + 1
    output_w = input_w - kw + 2 * pw + 1

    padding_h = np.max(output_h - 1 + kh - input_h, 0)
    padding_w = np.max(output_w - 1 + kw - input_w, 0)
    pad_top = padding_h // 2
    pad_bot = padding_h - pad_top
    pad_lft = padding_w // 2
    pad_rgt = padding_w - pad_lft

    padded_images = np.zeros((m, input_h + padding_h, input_w + padding_w))
    padded_images[:, pad_top:-pad_bot, pad_lft:-pad_rgt] = images

    output = np.zeros((m, output_h, output_w))

    for w in range(output_w):
        for h in range(output_h):
            output[:, h, w] = np.sum(
                kernel * padded_images[:, h:h + kh, w:w + kw],
                axis=(1, 2)
            )
    return output
