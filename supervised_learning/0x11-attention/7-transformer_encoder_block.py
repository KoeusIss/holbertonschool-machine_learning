#/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class Creates an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1) -> None:
        """Initializer

        Arguments:
            dm {int} -- Is a represention of the dimensionality of the model
            h {int} -- Is a represention of the number of head
            hidden {int} -- Is a representation of hidden fully connected layer

        Keyword Arguments:
            drop_rate {float} -- The drop rate (default: {0.1})
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Instance call

        Arguments:
            x {tf.Tensor} -- Is contains the input
            training {boolean} -- Is a representation of training

        Keyword Arguments:
            mask {None} -- Always none (default: {None})

        Returns:
            tf.Tensor -- Block's output
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        first = self.layernorm1(x + attention)

        hidden = self.dense_hidden(first)
        hidden = self.dense_output(hidden)
        second = self.dropout2(hidden, training=training)

        output = self.layernorm2(first + second)
        return output
