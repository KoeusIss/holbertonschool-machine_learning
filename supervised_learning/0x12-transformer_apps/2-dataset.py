#!/usr/bin/env python3
"""Transformer Application"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class loads and prepares a dataset for MT
    """
    def __init__(self) -> None:
        """Initializer
        """
        data_train, data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train
        )
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates subwords tokenizers for out dataset

        Arguments:
            data {tf.data.Dataset} -- Dataset of samples

        Returns:
            tuple -- Contains the tokenizer_pt and tokenizer_en
        """
        tok_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tok_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )
        return tok_pt, tok_en

    def encode(self, pt, en):
        """Enocdes a translation into tokens

        Arguments:
            pt {tf.Tensor} -- Contains the Portoguese sentence
            en {tf.Tensor} -- Contains the English sentence

        Returns:
            tuple -- Contains the encoded Portoguese sentence, and the encoded
            English sentence
        """
        pt_vsize = self.tokenizer_pt.vocab_size
        en_vsize = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vsize] +\
            self.tokenizer_pt.encode(pt.numpy()) + [pt_vsize + 1]
        en_tokens = [en_vsize] +\
            self.tokenizer_en.encode(en.numpy()) + [pt_vsize + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wraps the encoder into tensorflow operation

        Arguments:
            pt {tf.Tensor} -- Contains the Portoguese sentence
            en {tf.Tensor} -- Contains the English sentence

        Returns:
            tuple -- Contains the tensors of encoded Portoguese and English
            sentences
        """
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_lang.set_shape([None])
        en_lang.set_shape([None])
        return pt_lang, en_lang
