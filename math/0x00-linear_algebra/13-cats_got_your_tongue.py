#!/usr/bin/env python3
"""Matrices concatenation"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specefic axis

    Args:
        mat1 (list[list]): the first given matrix
        mat2 (list[list]): the second ginve matrix
        axis (int(0|1)): the given specific axis if axis == 0 so we need
        concatenates the two matrices along a column (verticaly) axis, else
        it must concatenated along the column axis (horizentaly)
    Returns:
        (list[list]|None): return the already concatenated new matrix, if the
        two matrix is uncompatible for concatenation, it should return None
    Example:
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6]]
        mat3 = [[7], [8]]
        cat_matrices2D(mat1, mat2) -> [[1, 2], [3, 4], [5, 6]
        cat_matrices2D(mat1, mat3, axis=1) -> [[1, 2, 7], [3, 4, 8]]

    """
    return np.concatenate((mat1, mat2), axis)
