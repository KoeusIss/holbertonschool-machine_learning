#!/usr/bin/env python3
"""Matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrices multiplication

    Args:
        mat1 (numpy.ndarray): the first given matrix
        mat2 (numpy.ndarray): the second given matrix
    Returns:
        (numpy.ndarray): returns a new numpy.ndarray where performs matrices
        multiplication
    Example:
        mat1 = [[1, 2],
                [3, 4],
                [5, 6]]
        mat2 = [[1, 2, 3, 4],
                [5, 6, 7, 8]]
        np_matmul(mat1, mat2) -> [[11, 14, 17, 20],
                                [23, 30, 37, 44],
                                [35, 46, 57, 68]]
    """
    return np.matmul(mat1, mat2)
