#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecodeBlock class creates decoder layer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1) -> None:
        """Initializer

        Arguments:
            dm {int} -- The output dimensionality
            h {int} -- The number of head
            hidden {int} -- The hidden units

        Keyword Arguments:
            drop_rate {float} -- Drop rate (default: {0.1})
        """
        super(DecoderBlock, self).__init__()
        self.mah1 = MultiHeadAttention(dm, h)
        self.mah2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )
        self.dense_ouput = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Instance call

        Arguments:
            x {tf.Tensot} -- Contains the input
            encoder_output {tf.Tensor} -- The encoder output
            training {Boolean} -- Indicates training for dropout
            look_ahead_mask {[type]} -- The mask applied for mha layer
            padding_mask {[type]} -- The mask applied for second mha layer

        Returns:
            tf.Tensor -- the decoder output
        """
        first_attention, _ = self.mah1(x, x, x, look_ahead_mask)
        first_attention = self.dropout1(first_attention, training=training)
        first_output = self.layernorm1(x + first_attention)

        second_attention, _ = self.mah2(
            encoder_output, encoder_output, first_output, padding_mask)
        second_attention = self.dropout2(second_attention, training=training)
        second_output = self.layernorm2(second_attention + first_output)

        third_output = self.dense_hidden(second_output)
        third_output = self.dense_ouput(third_output)
        third_output = self.dropout3(third_output, training=training)

        output = self.layernorm3(third_output + second_output)
        return output
