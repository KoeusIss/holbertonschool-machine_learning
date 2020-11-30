#!/usr/bin/env python3
""" Matrix shape """


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
