#!/usr/bin/env python3
"""Convolutions and pooling module"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images with channel

    Args:
        images (numpy.ndarray): Containing multiple gray scale images of shape
            (m, h, w, c) where m is the number of images, h is the height in
            pixels of the images and w is the width in pixels of images and
            c is the number of channel.
        kernel (numpy.ndarray): Containing the kernel for the convolution of
            shape (kh, kw) where kh is the height of the kernel and kw is the
            widht of the kernel.
        padding (tuple|str): containing (ph, pw) where ph is the padding along
            the height of the image, and the pw is the padding along the width
            or "same" or "valid".
        stride (tuple): Containing sh and sw where sh is the stride for the
            height of the image and sw is thte stride of the width.

    Returns:
        numpy.ndarray: Containing the convolved image.

    """
    m, input_h, input_w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == "valid":
        output_h = (input_h - kh) // sh + 1
        output_w = (input_w - kw) // sw + 1
        top, bot, lft, rgt = (0, 0, 0, 0)

    elif padding == "same":
        output_h = input_h // sh + 1
        output_w = input_w // sw + 1

        if kh % sh == 0:
            padding_h = max(kh // sh, 0)
        else:
            padding_h = max((kh - 1) // sh, 0)

        if kw % sw == 0:
            padding_w = max(kw // sw, 0)
        else:
            padding_w = max((kw - 1) // sw, 0)
        top, bot = (padding_h, padding_h)
        lft, rgt = (padding_w, padding_w)

    else:
        ph, pw = padding
        output_h = (input_h - kh + 2 * ph) // sh + 1
        output_w = (input_w - kw + 2 * pw) // sw + 1
        top, bot = (ph, ph)
        lft, rgt = (pw, pw)

    _images = np.pad(
        array=images,
        pad_width=((0, 0), (top, bot), (lft, rgt)),
        mode="constant",
        constant_values=0
    )

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                kernel * _images[
                    :,
                    i * sh:i * sh + kh,
                    j * sw:j * sw + kw
                ],
                axis=(1, 2)
            )
    return output
