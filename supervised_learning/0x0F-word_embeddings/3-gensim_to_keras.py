#!/usr/bin/env pyhton3
"""Word Embeddings module"""


def gensim_to_keras(model):
    """Convert a genesim model to keras Embedding layer

    Arguments:
        model {genesim word2vec} -- Genesim word2vec model

    Returns:
        keras.model -- Keras model
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
