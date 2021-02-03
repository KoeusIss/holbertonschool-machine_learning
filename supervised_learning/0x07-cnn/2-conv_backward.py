#!/usr/bin/env python3
"""Convolution Neural Nerwork module"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back-propagation over a convolutional layer of a neural network

    Args:
        dZ (numpy.ndarra): Is containing the partial derivatives with respect
            to the unactivatted ouput of the convolutional layer with shape
            (m, h_new, w_new, c_new) where m is the number of input data point,
            h_new and w_new is the height and the width of the output, and
            c_new is the number of channels in the output.
        A_prev (numpy.ndarray): Is containing the output of te previous layer
            with shape (m, h_prev, w_prev, c_prev) where m is the number of
            input data points, h_prev and w_prec is the height and width of the
            previous output, and c_prev the number of cahnnels in previous.
        W (numpy.ndarray): Is containing the kernels for the convolution with
            shape (kh, kw, c_prev, c_new)
        b (numpy.ndarray): Is containing the the biases.
        padding (str): Is idicates either applie a `same` or `valid`
            convolution.
        stride (tuple): Is containg the height and width applied stride.

    Returns:
        numpy.ndarray: The partial derivative with respect to previous layer's
        activation, The kernels and biases

    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    padding_h, padding_w = (0, 0)

    if padding == "same":
        padding_h = int(np.ceil((sh * (h_prev - 1) - h_prev + kh) / 2))
        padding_w = int(np.ceil((sw * (w_prev - 1) - w_prev + kw) / 2))

    _A_prev = np.pad(
        array=A_prev,
        pad_width=((0,), (padding_h,), (padding_w,), (0,)),
        mode="constant",
        constant_values=0
    )
    dA_prev = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    dA_prev[
                        n, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                    ] += W[:, :, :, c] * dZ[n, i, j, c]
                    dW[:, :, :, c] += _A_prev[
                        n, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                    ] * dZ[n, i, j, c]
                    db[:, :, :, c] += dZ[n, i, j, c]
    return dA_prev, dW, db
