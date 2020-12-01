#!/usr/bin/env python3
"""Matrices concatenation"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specefic axis

    Args:
        mat1 (numpy.ndarray): the first given matrix
        mat2 (numpy.ndarray): the second ginve matrix
        axis (int(0|1)): the given specific axis if axis == 0 so we need
        concatenates the two matrices along a column (verticaly) axis, else
        it must concatenated along the column axis (horizentaly)
    Returns:
        (numpy.ndarray): return the already concatenated new matrix, if the
        two matrix is uncompatible for concatenation, it should return None
    Example:
        mat1 = np.array([[11, 22, 33], [44, 55, 66]])
        mat2 = np.array([[1, 2, 3], [4, 5, 6]])
        np_cat(mat1, mat2) -> [[11 22 33]
                                [44 55 66]
                                [ 1  2  3]
                                [ 4  5  6]]

    """
    return np.concatenate((mat1, mat2), axis)
