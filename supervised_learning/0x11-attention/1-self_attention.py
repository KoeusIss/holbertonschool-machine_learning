#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention class -- Calculate attention for machine translation
    based on this paper https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, units) -> None:
        """Initializer

        Arguments:
            units {int} -- Is representing the number of hidden units in the
            alignment model
        """
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, s_prev, hidden_states):
        """Instance Call

        Arguments:
            s_prev {tf.Tensor} -- Is containing the previous decoder hidden
            state of shape (batch, units).
            hidden_states {tf.Tensor} -- Is Contatining the outputs of the
            of shape (batch, input_seq_len, units)

        Returns:
            tuple -- Contains a tf.Tensor contains context vector for the
            decoder of shape (batch, units), and tf.Tensor contains the
            attention weights of shape (batch, input_seq_len, 1).
        """
        s_prev_time = tf.expand_dims(s_prev, 1)
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev_time) + self.U(hidden_states)
            )
        )
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
