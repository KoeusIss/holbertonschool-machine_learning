#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder class -- Encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch) -> None:
        """Initializer

        Arguments:
            vocab {int} -- Is representing the size of the input vocabulary
            embedding {int} -- Is representing the dimensionality of embedding
            units {int} -- Is the number of hidden units in the RNN cell
            batch {int} -- Is representing the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self) -> tf.Tensor:
        """Initializes the hidden states of RNN cell to a tensor zeros

        Returns:
            tf.Tensor -- Containing the initial hidden state
        """
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial) -> tuple:
        """Instance Call

        Arguments:
            x {tf.Tensor} -- Is a tensor of shape (batch, input_seq_len)
            containing the input to the encoder layers as word indices within
            the vocabulary
            initial {tf.Tensor} -- Is a tensor of shape (batch, units)
            containing the initial hidden state

        Returns:
            tuple -- Is containing a tensor of shape
            (batch, input_seq_len, units) contains the output of the encoder,
            and a tensor of shape (batch, units) containing the last hidden
            state of the encoder
        """
        embedded = self.embedding(x)
        return self.gru(inputs=embedded, initial_state=initial)
