#!/usr/bin/env python3
"""NLP Metrics"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for sentence

    Arguments:
        references {list} -- Is a list of reference translations
        sentence {list} -- Contains the model proposed sentence

    Returns:
        float -- The unigram BLEU score
    """
    numerator = 0
    flatten_ref = set([word for ref in references for word in ref])

    for word in flatten_ref:
        if word in sentence:
            numerator += 1
    precision = numerator / len(sentence)

    best_match = None
    for ref in references:
        if best_match is None:
            best_match = ref
        best_diff = abs(len(best_match) - len(sentence))
        if abs(len(ref) - len(sentence)) < best_diff:
            best_match = ref

    if len(sentence) > len(best_match):
        brevity_penality = 1
    else:
        brevity_penality = np.exp(1 - len(best_match) / len(sentence))

    bleu_score = brevity_penality * precision
    return bleu_score
