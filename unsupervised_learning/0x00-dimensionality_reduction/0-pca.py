#!/usr/bin/env python3
"""Dimensionality reduction module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset

    Args:
        X (np.ndarray): of shape (n, d) where n is the number of data points
            ans d is the number of dimensions.
        var (float): is the fraction of variance that the PCA transformation
            should maintain.

    Retruns:
        np.ndarray: the weights matrix of shape (d, nd) where nd is the
            dimension of transformed X

    """
    U, s, V = np.linalg.svd(X)
    cumulated = np.cumsum(s)
    percentage = cumulated / np.sum(s)
    r = np.argwhere(percentage >= var)[0, 0]
    return V[:r + 1].T
