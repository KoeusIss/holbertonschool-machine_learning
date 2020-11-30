#!/usr/bin/env python3
""" Arrays concatenation """


def cat_arrays(arr1, arr2):
    """Concatenates two array of ints/floats

    Args:
        arr1 (list): the first given list
        arr2 (list): the second given list
    Returns:
        (list): A new list contains the two given arrays concatenated in
        single list
    Example:
        arr1 = [1, 2, 3, 4, 5]
        arr2 = [6, 7, 8]
        cat_arrays(arr1, arr2) -> [1, 2, 3, 4, 5, 6, 7, 8]

    """
    return arr1 + arr2
