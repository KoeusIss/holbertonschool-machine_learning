#!/usr/bin/env python3
""" Matrix transpose """


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix by flipping between columns and rows
        matrix (list[list]): the given 2D matrix
    """
    matrix_t = []
    for i in range(len(matrix[0])):
        matrix_t_row = []
        for j in range(len(matrix)):
            matrix_t_row.append(matrix[j][i])
        matrix_t.append(matrix_t_row)
    return matrix_t
