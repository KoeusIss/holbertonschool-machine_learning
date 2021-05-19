#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention

    Arguments:
        Q {tf.Tensor} -- Contains the query matrix
        K {tf.Tensor} -- Contains the key matrix
        V {tf.Tensor} -- Contains the value matrix

    Keyword Arguments:
        mask {tf.Tensor} -- Contains the optional mask (default: {None})

    Returns:
        tuple -- contains tf.Tensor for scaled dot product attention, and
        tf.Tensor contains the attention weights
    """
    matmul_QK = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul_QK / tf.sqrt(dk)
    if mask is not None:
        scaled += mask
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
