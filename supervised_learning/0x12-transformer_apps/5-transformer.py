#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer
    """
    def get_angles(pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dm))
        return pos * angle_rates

    position = np.arange(max_seq_len)
    PE = get_angles(position[:, np.newaxis], np.arange(dm)[np.newaxis, :])
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    return PE


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention
    """
    matmul_QK = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = matmul_QK / tf.sqrt(dk)
    if mask is not None:
        scaled += mask
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class perform multi-head attention
    """
    def __init__(self, dm, h) -> None:
        """Initializer
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = self.dm // self.h
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        """Instance call
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        def split_heads(x):
            """Splits inputs into heads
            """
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        output = self.linear(output)
        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class Creates an encoder block for a transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1) -> None:
        """Initializer
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
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        first = self.layernorm1(x + attention)

        hidden = self.dense_hidden(first)
        hidden = self.dense_output(hidden)
        second = self.dropout2(hidden, training=training)

        output = self.layernorm2(first + second)
        return output


class DecoderBlock(tf.keras.layers.Layer):
    """DecodeBlock class creates decoder layer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1) -> None:
        """Initializer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Instance call
        """
        first_attention, _ = self.mha1(x, x, x, look_ahead_mask)
        first_attention = self.dropout1(first_attention, training=training)
        first_output = self.layernorm1(x + first_attention)

        second_attention, _ = self.mha2(
            first_output, encoder_output, encoder_output, padding_mask)
        second_attention = self.dropout2(second_attention, training=training)
        second_output = self.layernorm2(second_attention + first_output)

        third_output = self.dense_hidden(second_output)
        third_output = self.dense_output(third_output)
        third_output = self.dropout3(third_output, training=training)
        output = self.layernorm3(third_output + second_output)
        return output


class Encoder(tf.keras.layers.Layer):
    """Encoder class
    """
    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """Initializer
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
        """
        input_seq_len = x.shape[1]
        embedded = self.embedding(x)
        scaled = embedded * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positioned = scaled + self.positional_encoding[:input_seq_len, :]
        x = self.dropout(positioned, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder class
    """
    def __init__(
        self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1
    ) -> None:
        """Initializer
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(self.N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Instance call
        """
        input_seq_len = x.shape[1]
        embedded = self.embedding(x)
        scaled = embedded * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positioned = scaled + self.positional_encoding[:input_seq_len, :]
        x = self.dropout(positioned, training=training)

        for block in self.blocks:
            x = block(
                x, encoder_output, training, look_ahead_mask, padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Transformer class
    """
    def __init__(
        self, N, dm, h, hidden, input_vocab, target_vocab,
        max_seq_input, max_seq_target, drop_rate=0.1
    ):
        """Initializer
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self, inputs, target, training, encoder_mask,
        look_ahead_mask, decoder_mask
    ):
        """Instance call
        """
        encoded = self.encoder(inputs, training, encoder_mask)
        decoded = self.decoder(
            target, encoded, training, look_ahead_mask, decoder_mask
        )
        output = self.linear(decoded)
        return output
