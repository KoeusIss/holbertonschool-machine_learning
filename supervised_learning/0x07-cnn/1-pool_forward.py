#!/usr/bin/env python3
"""Convolution Neural Network module"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network

    Args:
        A_prev (numpy.ndarray): Is containing the output of the previous layer
            of shape (m, h_prev, w_prev, c_prev) where m is the number of input
            data point, h_prev and w_prev is the height and width in pixel of
            the previous output and c_prev is the number of channel of previous
            output.
         kernel_shape (tuple): Is containing the shapes for the convolution of
            shape (kh, kw) where kh and kw is the height of the pooling kernel.
        stride (tuple): Is containing the strides for the convolution of
            shape (sh, sw) where sh is the stride for the height and sw for
            the width
        mode (str): Is indicates either `max` or `avg` pooling.

    Returns:
        numpy.ndarray: The output of the convolutional layer

    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h_prev - kh) / sh + 1)
    output_w = int((w_prev - kw) / sw + 1)

    if mode == "max":
        pool_operation = np.max
    else:
        pool_operation = np.average

    output = np.zeros((m, output_h, output_w, c_prev))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j, :] = pool_operation(
                A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                axis=(1, 2)
            )
    return output
