#!/usr/bin/env python3
"""Word Embeddings module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Creates a TFIDF embedding matrix

    Arguments:
        sentences {list[str]} -- List of sentences to analyze

    Keyword Arguments:
        vocab {list} -- List of the vocabulary words (default: {None})

    Returns:
        tuple(np.ndarray, list) -- Containing the embeddings matrix, list of
        features
    """
    tfids_vector = TfidfVectorizer(vocabulary=vocab)
    embeddings = tfids_vector.fit_transform(sentences)
    features = tfids_vector.get_feature_names()
    return embeddings.toarray(), features
