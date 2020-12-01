#!/usr/bin/python3
""" Matrices multiplication """


def mat_mul(mat1, mat2):
    """Mutiplies two matrices

    Args:
        mat1 (list[list]): the first given matrix
        mat2 (list[list]): the second given matrix
    Returns:
        (list[list]|None): returns a new list where performs matrices
        multiplication, otherwise when the number of columns of the
        first matrix is different from the number of rows of the
        second return None
    Example:
        mat1 = [[1, 2],
                [3, 4],
                [5, 6]]
        mat2 = [[1, 2, 3, 4],
                [5, 6, 7, 8]]
        mat_mul(mat1, mat2) -> [[11, 14, 17, 20],
                                [23, 30, 37, 44],
                                [35, 46, 57, 68]]

    """
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        result_row = []
        for j in range(len(mat2[0])):
            row_sum = 0
            for k in range(len(mat1[0])):
                row_sum += mat1[i][k] * mat2[k][j]
            result_row.append(row_sum)
        result.append(result_row)
    return result
