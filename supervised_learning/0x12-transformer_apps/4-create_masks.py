#!/usr/bin/env python3
"""Transformer application"""
import tensorflow as tf


def create_padding_mask(seq):
    """Creates padding mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Creates look ahead mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """Creates masks for Training/Validation

    Arguments:
        inputs {tf.Tensor} -- Of shape (batch_size, seq_len_in) that contains
        the input sentence
        target {tf.Tensor} -- of shape (batch_size, seq_len_out) that contains
        the target sentence

    Returns:
        tuple -- Contains the masks
    """
    size = tf.shape(target)[1]
    encoder_mask = create_padding_mask(inputs)

    decoder_mask = create_padding_mask(inputs)

    lam = create_look_ahead_mask(size)
    tpm = create_padding_mask(target)
    combined_mask = tf.maximum(tpm, lam)

    return encoder_mask, combined_mask, decoder_mask
