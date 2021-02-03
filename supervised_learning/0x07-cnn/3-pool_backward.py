#!/usr/bin/env python3
"""Convolution Neural Nerwork module"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back-propagation over a convolutional layer of a neural network

    Args:
        dA (numpy.ndarra): Is containing the partial derivatives with respect
            to the unactivatted ouput of the convolutional layer with shape
            (m, h_new, w_new, c) where m is the number of input data point,
            h_new and w_new is the height and the width of the output, and
            c is the number of channels in the output.
        A_prev (numpy.ndarray): Is containing the output of te previous layer
            with shape (m, h_prev, w_prev, c) where m is the number of
            input data points, h_prev and w_prec is the height and width of the
            previouj output, and c_prev the number of cahnnels in previous.
        kernel_shape (tuple): Is containing the kernels for the convolution of
            shape (kh, kw)
        stride (tuple): Is containg the height and width applied stride.
        mode (str): Is either `max` or `avg`

    Returns:
        numpy.ndarray: The partial derivative with respect to previous layer's
        activation

    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c):
                    if mode == "max":
                        tmp = A_prev[
                            n, i * sh:i * sh + kh, j * sw:j * sw + kw, k
                        ]
                        mask = np.where(tmp == np.max(tmp), 1, 0)
                        dA_prev[
                            n, i * sh:i * sh + kh, j * sw:j * sw + kw, k
                        ] += dA[n, i, j, k] * mask
                    else:
                        avg = dA[n, i, j, k] / (kw * kh)
                        dA_prev[
                            n, i * sh:i * sh + kh, j * sw:j * sw + kw, k
                        ] += np.ones(kernel_shape) * avg
    return dA_prev
