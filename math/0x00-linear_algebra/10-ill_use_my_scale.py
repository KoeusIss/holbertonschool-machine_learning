#!/usr/bin/env python3
""" Scale a matrix """


def np_shape(matrix):
    """Calculates the shape of a numpy.ndarray

    Args:
        matrix (numpy.ndarray): the given matrix
    Returns:
        (tuple): returns a tuple of integers indicates matrix's shape
    Example:
        mat1 = np.array([1, 2, 3, 4, 5, 6])
        np_shape(mat1) -> (6,)

    """
    return matrix.shape
