#!/usr/bin/env python3
"""Matrices addition"""


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


def add_matrices(mat1, mat2):
    """Adds two matrices in a new matrix

    Args:
        mat1 (list[list, ...]): first given n dimension list
        mat2 (list[list, ...]): second given n dimension list
    Returns:
        (list[list, ...]: returns a matrix with n dimension)

     """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(matrix_shape(mat1)) > 1:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
    else:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
