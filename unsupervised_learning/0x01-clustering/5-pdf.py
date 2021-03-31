#!/usr/bin/env python3
"""Clustering module"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function for Guassian distribution

    Arguments:
        X {np.ndarray} -- Containing the data point
        m {np.ndarray} -- Contating the mean of data point
        S {np.ndarray} -- Contating the covariance matrix

    Returns:
        np.ndarray -- The probability density function.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    cst = 1 / (np.sqrt((2 * np.pi)**d * S_det))
    val = np.einsum('...i, ij, ...j -> ...', X - m, S_inv, X - m)
    pdf = cst * np.exp((-1/2) * val)
    return np.maximum(pdf, 1e-300)
