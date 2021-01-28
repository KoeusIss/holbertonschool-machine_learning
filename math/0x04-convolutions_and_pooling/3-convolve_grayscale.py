#!/usr/bin/env python3
"""Convolutions and pooling module"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images

    Args:
        images (numpy.ndarray): Containing multiple gray scale images of shape
            (m, h, w) where m is the number of images, h is the height in
            pixels of the images and w is the width in pixels of images.
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
        output_h = int(np.floor((input_h - kh + 1) / sh))
        output_w = int(np.floor((input_w - kw + 1) / sw))
        _images = images

    elif padding == "same":
        output_h = int(np.ceil(input_h / sh))
        output_w = int(np.ceil(input_w / sw))
        if input_h % sh == 0:
            padding_h = np.max(kh - sh, 0)
        else:
            padding_h = np.max(kh - (input_h % sh), 0)
        if input_w % sw == 0:
            padding_w = np.max(kh - sh, 0)
        else:
            padding_w = np.max(kh - (input_h % sh), 0)
        _images = np.pad(
            array=images,
            pad_width=((0,), (padding_h,), (padding_w,)),
            mode="constant",
            constant_values=0
        )

    else:
        ph, pw = padding
        output_h = int(np.ceil((input_h - kh + 2 * ph + 1) / sh))
        output_w = int(np.ceil((input_w - kw + 2 * pw + 1) / sw))
        padding_h = 2 * ph
        padding_w = 2 * pw
        _images = np.pad(
            array=images,
            pad_width=((0,), (padding_h,), (padding_w,)),
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
