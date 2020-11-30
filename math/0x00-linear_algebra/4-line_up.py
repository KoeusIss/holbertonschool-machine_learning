#!/usr/bin/env python3
""" Vector addition """


def add_arrays(arr1, arr2):
    """
    Adds two arrays and return a new list, if the two arrays are
    not compatible length, then return None
        arr1 (list): the first given array
        arr2 (list): the second given array
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[idx] + arr2[idx] for idx in range(len(arr1))]
