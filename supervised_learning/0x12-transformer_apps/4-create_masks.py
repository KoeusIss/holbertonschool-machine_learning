#!/usr/bin/env python3
"""Transformer application"""
import tensorflow as tf


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
    batch_size, seq_len_out = target.shape

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.neaxis, :]

    combined_mask = 1 - tf.linalg.band_part(
        tf.ones((batch_size, 1, seq_len_out, seq_len_out)), -1, 0
    )

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = encoder_mask[:, tf.newaxis, tf.neaxis, :]

    return encoder_mask, combined_mask, decoder_mask
