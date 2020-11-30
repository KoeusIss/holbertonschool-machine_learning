#!/usr/bin/env python3
""" Matrix shape """


def matrix_shape(matrix):
    """
    Find the shape of a given matrix by wrapping a recursive function to avoid
    data persistance in the size list.
        matrix (list[list]): the given matrix
    """
    def shape(m, size=[]):
        """
        Finds the shape of a matrix m recursively by appneding each inner
        list length in the size list
            m (list[list]): the given matrix
            size (list): the result shape lsit
        """
        if type(m) is not list:
            return size
        else:
            size.append(len(m))
            return shape(m[0], size)
    return shape(matrix)
