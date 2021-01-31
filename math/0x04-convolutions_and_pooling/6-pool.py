#!/usr/bin/env python3
"""Convolutions and pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images

    Args:
        images (numpy.ndarray): Containing multiple images of shape
            (m, h, w, c) where m is the number of images, h is the height on
            pixels, w is the width on pixels and c is the number of channel.
        kernel_shape (tuple): Containing the kernel shape of shape (kh, kw)
            where kh is the kernel height on pixels, kw is the kernel widht on
            pixlels.
        stride (tuple): Containing the strides of shape (sh, sw) where sh is
            the stride of the height and sw is the stride of the width.
        mode (str): Indicates the type of pooling, `max` indicates max pooling
            `avg` indicates the average pooling

    Returns:
        numpy.ndarray: Containing the pooled images

    """
    m, input_h, input_w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int(np.ceil((input_h - kh) / sh + 1))
    output_w = int(np.ceil((input_w - kw) / sw + 1))

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == "max":
                output[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
            else:
                output[:, i, j, :] = np.average(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
    return output
