#!/usr/bin/env python3
""" Matrix transpose """


def np_transpose(matrix):
    """Calculates the matrix transpose

    Args:
        matrix (numpy.ndarray): a given matrix
    Returns:
        (numpy.ndarray): returns the transpose of the matrix
    Example:
        mat1 = np.array([1, 2, 3, 4, 5, 6])
        np_transpose(mat1)

    """
    return matrix.T
