#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class decodes for machine translation
    """
    def __init__(self, vocab, embedding, units, batch) -> None:
        """Initializer

        Arguments:
            vocab {int} -- Is representing the size of the input vocabulary
            embedding {int} -- Is representing the dimensionality of embedding
            units {int} -- Is the number of hidden units in the RNN cell
            batch {int} -- Is representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Instance call

        Arguments:
            x {tf.Tensor} -- Is a tensor of shape (batch, 1) contains the
            previous word in the target squence.
            s_prev {tf.Tensor} -- Is a tensor of shape (batch, units) contains
            the previous decoder hidden state
            hidden_states {tf.Tensor} -- Contains the outputs of the encoder
            of shape (btach, input_seq_len, units).

        Returns:
            tuple -- Contains a tf.Tensor of shape (batch, vocab) of the output
            word as a one hot vector, and tf.Tensor of shape (batch, units)
            contains the new decoder hidden state
        """
        self_attention = SelfAttention(self.units)
        context, _ = self_attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        inputs = tf.concat([context, x], axis=-1)
        output, state = self.gru(inputs=inputs)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, state
