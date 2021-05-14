#!/usr/bin/env python3
"""NLP Metrics module"""
import numpy as np


def n_gram(sentence, n):
    """Tokenize sentence into grams

    Arguments:
        sentence {list} -- Is containing the sting
        n {int} -- Is the prefered n-gram

    Returns:
        list -- Containg n-gram sentence
    """
    step = n - 1

    result = sentence[:-step]
    for i in range(len(result)):
        for j in range(step):
            result[i] += ' ' + sentence[i + 1 + j]
    return result


def ngram_bleu(references, sentence, n):
    """Calucluates the n-gram BLEU score

    Arguments:
        references {list} -- Containg a list of string sentence reference
        sentence {list} -- Contain the model candidate
        n {int} -- The number of prefered grams

    Returns:
        float -- The n-gram BLEU score
    """
    c = len(sentence)
    rs = [len(r) for r in references]

    sentence = n_gram(sentence, n)
    references = list(map(lambda ref: n_gram(ref, n), references))
    flatten_ref = set([gram for ref in references for gram in ref])

    numerator = 0
    for gram in flatten_ref:
        if gram in sentence:
            numerator += 1
    precision = numerator / len(sentence)

    best_match = None
    for i, ref in enumerate(references):
        if best_match is None:
            best_match = ref
            r_idx = i
        best_diff = abs(len(best_match) - len(sentence))
        if abs(len(ref) - len(sentence)) < best_diff:
            best_match = ref
            r_idx = i

    r = rs[r_idx]
    if c > r:
        brevity_penality = 1
    else:
        brevity_penality = np.exp(1 - r / c)

    # exp(log(x)) return x, its a placeholder for weights ;)
    bleu_score = brevity_penality * np.exp(np.log(precision))
    return bleu_score
