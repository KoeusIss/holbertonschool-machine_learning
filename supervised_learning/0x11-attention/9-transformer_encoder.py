#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1) -> None:
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding =
