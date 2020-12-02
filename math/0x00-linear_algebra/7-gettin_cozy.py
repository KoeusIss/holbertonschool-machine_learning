#!/usr/bin/env python3
""" Matrices concatenation """


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

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
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        first = [[i for i in row1] for row1 in mat1]
        second = [[j for j in row2] for row2 in mat2]
        return first + second
    else:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
