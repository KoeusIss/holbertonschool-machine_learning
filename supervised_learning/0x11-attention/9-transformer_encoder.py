#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder class
    """
    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """Initializer

        Arguments:
            N {int} -- The number of blocks
            dm {int} -- Model dimensionality
            h {int} -- The number of heads
            hidden {int} -- The number of hidden units
            input_vocab {int} -- The vocabulary length
            max_seq_len {int} -- Maximum sequence length

        Keyword Arguments:
            drop_rate {float} -- Drop rate (default: {0.1})
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Instance call

        Arguments:
            x {tf.Tensor} -- Input tensor
            training {Boolean} -- Indicates training
            mask {[type]} -- The mask

        Returns:
            tf.Tensor -- The output of encoder layer
        """
        input_seq_len = x.shape[1]
        embedded = self.embedding(x)
        scaled = embedded * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positioned = scaled + self.positional_encoding[:input_seq_len, :]
        x = self.dropout(positioned, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
