#!/usr/bin/env python3
"""Advanced linear algebra module"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definetness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if any(len(row) != len(matrix) for row in matrix):
        return None

    if len(matrix.shape) < 2:
        return None

    eivals = np.linalg.eig(matrix)[0]
    if all(eivals >= 0):
        if any(eivals == 0):
            return 'Positive semi-definite'
        return 'Positive definite'
    if all(eivals <= 0):
        if any(eivals == 0):
            return 'Negative semi-definite'
        return 'Negative definite'
    return 'Indefinite'
