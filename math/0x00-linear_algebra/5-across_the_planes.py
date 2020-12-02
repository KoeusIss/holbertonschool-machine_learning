#!/usr/bin/env python3
""" Matrices addition """


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
    if len(mat1) != len(mat2):
        return None
    try:
        if len(mat1[0]) != len(mat2[0]):
            return None
    except:
        None
    result = []
    for i in range(len(mat1)):
        result_row = []
        for j in range(len(mat1[0])):
            result_row.append(mat1[i][j] + mat2[i][j])
        result.append(result_row)
    return result
