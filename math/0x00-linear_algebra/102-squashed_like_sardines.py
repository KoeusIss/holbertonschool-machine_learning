#!/usr/bin/env python3
"""Matrices concatenation"""


def matrix_shape(matrix):
    """Finds the shape of a matrix"""
    def shape(m, size=None):
        """Finds the shape of matrix recursively"""
        if size is None:
            size = []
        if type(m) is not list:
            return size
        else:
            size.append(len(m))
            return shape(m[0], size)
    return shape(matrix)


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates matrices"""
    if len(matrix_shape(mat1)) != len(matrix_shape(mat2)):
        return None
    if axis == 0:
        if isinstance(mat1, list) and isinstance(mat1[0], list):
            first = [[i for i in row1] for row1 in mat1]
            second = [[j for j in row2] for row2 in mat2]
        else:
            first = mat1.copy()
            second = mat2.copy()
        return first + second
    else:
        return [cat_matrices(r1, r2, axis - 1) for r1, r2 in zip(mat1, mat2)]
