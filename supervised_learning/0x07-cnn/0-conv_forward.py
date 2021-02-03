#!/usr/bin/env python3
"""Convolutional Neural Network module"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural
    Network.

    Args:
        A_prev (numpy.ndarray): Is containing the output of the previous layer
            of shape (m, h_prev, w_prev, c_prev) where m is the number of input
            data point, h_prev and w_prev is the height and width in pixel of
            the previous output and c_prev is the number of channel of previous
            output.
        W (numpy.ndarray): Is containing the kernels for the convolution of
            shape (kh, kw, c_prev, c_new) where kh and kw is the height and the
            width of the kernels, c_prev is the of channels of the previous
            layer and c_new is the number of channels in the output.
        b (numpy.ndarray): Is containing the biases of applied to the
            convolution of shape (1, 1, 1, c_new) where c_new is the number
            of channels in the ouput.
        activation (str): Indicates the activation function applied to the
            convolution.
        padding (str): Indicates is either `same` or `valid` used padding.
        stride (tuple): Is containing the strides for the convolution of
            shape (sh, sw) where sh is the stride for the height and sw for
            the width

    Returns:
        numpy.ndarray: The output of the convolutional layer

    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        output_h = int((h_prev - kh) / sh + 1)
        output_w = int((w_prev - kw) / sw + 1)
        padding_h = 0
        padding_w = 0
    else:
        output_h = h_prev
        output_w = w_prev
        padding_h = int(ceil(sh * (output_h - 1) - h_prev + kh) / 2)
        padding_w = int(ceil(sw * (output_w - 1) - w_prev + kw) / 2)

    A_padded = np.pad(
        array=A_prev,
        pad_width=((0,), (padding_h,), (padding_w,), (0,)),
        mode="constant",
        constant_values=0
    )
    output = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                output[:, i, j, k] = np.sum(
                    W[:, :, :, k] *
                    A_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2, 3)
                )
    return output + b
