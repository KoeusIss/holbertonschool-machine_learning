#/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        self.dm = dm
        self.h = h
        self.hidden = hidden
        self.mha = MultiHeadAttention(self.dm, self.h)
        self.dense_hidden = tf.keras.layers.Dense(
            self.hidden,
            activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(self.dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    call(self, x, training, mask=None):

