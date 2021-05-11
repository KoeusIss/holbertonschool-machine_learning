#!/usr/bin/env python3
"""Word Embeddings module"""
from sklearn.feature_extraction.text import CountVectorizer


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
    cont_vector = CountVectorizer()
    embeddings = cont_vector.fit_transform(sentences, vocab)
    features = cont_vector.get_feature_names()
    return embeddings.toarray(), features
