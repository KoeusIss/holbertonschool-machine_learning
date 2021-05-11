#!/usr/bin/env python3
"""Word Embeddings module"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix

    Arguments:
        sentences {list[str]} -- List of sentences to analyze

    Keyword Arguments:
        vocab {list} -- List of the vocabulary words (default: {None})

    Returns:
        tuple(np.ndarray, list) -- Containing the embeddings matrix, list of
        features
    """
    PONCTUATION = ".,;!?"
    if vocab is None:
        words = [word for sentence in sentences for word in sentence.split()]
        vocab = sorted(set(word.lower().strip(PONCTUATION) for word in words))

    feats = vocab
    s = len(sentences)
    f = len(feats)
    embds = np.zeros((s, f), dtype='int')
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(feats):
            if word in sentence.lower():
                embds[i, j] = 1
    return embds, feats
