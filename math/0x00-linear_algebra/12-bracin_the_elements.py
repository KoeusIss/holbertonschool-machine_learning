#!/usr/bin/env python3
"""Matrices operations"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, substraction, multiplication
    and division.

    Args:
        mat1 (numpy.ndarray): the first given matrix
        mat2 (numpy.ndarray): the second given matrix
    Returns:
        (tuple): contains the element-wise sum, difference, product,
        and quotient.
    Example:
        mat1 = np.array([[11, 22, 33], [44, 55, 66]])
        mat2 = np.array([[1, 2, 3], [4, 5, 6]]
        np_elementwise(mat1, mat2) -> (add, sub, mul, div) ~ ([numpy.ndarray])

    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
