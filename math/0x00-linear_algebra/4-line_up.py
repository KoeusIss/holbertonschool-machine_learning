#!/usr/bin/env python3
""" Vector addition """


def add_arrays(arr1, arr2):
    """Adds two arrays and return a new list, if the two arrays are
    not compatible length, then return None

    Args:
        arr1 (list): the first given array
        arr2 (list): the second given array
    Returns:
        (list): contains the sum of the two arrays element by element
    Example:
        arr1 = [1, 2, 3, 4]
        arr2 = [5, 6, 7, 8]
        add_arrays(arr1, arr2) -> [6, 8, 10, 12]

    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[idx] + arr2[idx] for idx in range(len(arr1))]
