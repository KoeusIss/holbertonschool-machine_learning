#!/usr/bin/env python3
"""Convolutions and pooling module"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images with channels

    Args:
        images (numpy.ndarray): Containing multiple gray scale images of shape
            (m, h, w, c) where m is the number of images, h is the height in
            pixels of the images and w is the width in pixels of images and
            c is the number of channels.
        kernel (numpy.ndarray): Containing the kernel for the convolution of
            shape (kh, kw, kc) where kh is the height of the kernel and kw is
            the widht of the kernel and kc is the number of channel.
        padding (tuple|str): containing (ph, pw) where ph is the padding along
            the height of the image, and the pw is the padding along the width
            or "same" or "valid".
        stride (tuple): Containing sh and sw where sh is the stride for the
            height of the image and sw is thte stride of the width.

    Returns:
        numpy.ndarray: Containing the convolved image.

    """
    m, input_h, input_w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == "valid":
        output_h = (input_h - kh) // sh + 1
        output_w = (input_w - kw) // sw + 1
        top, bot, lft, rgt, fnt, bck = (0, 0, 0, 0, 0, 0)

    elif padding == "same":
        output_h = input_h
        output_w = input_w

        padding_h = int(((input_h - 1) * sh + kh - input_h) / 2) + 1
        padding_w = int(((input_w - 1) * sw + kw - input_w) / 2) + 1

        top = padding_h
        bot = padding_h
        lft = padding_w
        rgt = padding_w

    else:
        ph, pw = padding
        output_h = (input_h - kh + 2 * ph) // sh + 1
        output_w = (input_w - kw + 2 * pw) // sw + 1
        top, bot = (ph, ph)
        lft, rgt = (pw, pw)

    _images = np.pad(
        array=images,
        pad_width=((0, 0), (top, bot), (lft, rgt), (0, 0)),
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
                axis=(1, 2, 3)
            )
    return output
