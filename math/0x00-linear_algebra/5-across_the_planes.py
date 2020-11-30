#!/usr/bin/env python3
""" Matrices addition """


def matrix_shape(matrix):
    """Finds the shape of a given matrix by wrapping a recursive function
    to avoid data persistance in the size list.

    Args:
        matrix (list[list]): the given matrix
    Returns:
        (list): a list contain the shape of the given matrix.
    Example:
        mat1 = [[1, 2], [3, 4]]
        matrix_shape(mat1) -> [2, 2]

    """
    def shape(m, size=[]):
        """Finds the shape of a matrix m recursively by appneding each inner
        list length in the size list

        Args:
            m (list[list]): the given matrix
            size (list): the result shape lsit

        """
        if type(m) is not list:
            return size
        else:
            size.append(len(m))
            return shape(m[0], size)
    return shape(matrix)


def add_matrices2D(mat1, mat2):
    """Adds two matrices in a new 2D matrix

    Args:
        mat1 (list[list]): A given 2D matrix
        mat2 (list[list]): A given 2D matrix
    Returns:
        (list[list])|None: A 2D matrix contain the addition of the
        two given matrix element by element or return None if the shape of
        the two matrices are not compatible
    Example:
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        add_matrices2D(mat1, mat2) -> [[6, 8], [10, 12]]

    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        result_row = []
        for j in range(len(mat1[0])):
            result_row.append(mat1[i][j] + mat2[i][j])
        result.append(result_row)
    return result
