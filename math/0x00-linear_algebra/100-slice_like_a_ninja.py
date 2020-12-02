#!/usr/bin/env python3
"""Slice like a Ninja"""


def get_slice(slice_tpl) -> slice:
    """Casts a tuple to slice

    Args:
        slice_tpl (tuple): a given tuple representing the slice to make
    Returns:
        (slice): returns a slice object containing start, stop and step
        of the slicing
    """
    try:
        start = slice_tpl[0]
    except Exception:
        start = None
    try:
        stop = slice_tpl[1]
    except Exception:
        stop = None
    try:
        step = slice_tpl[2]
    except Exception:
        step = None

    return slice(start, stop, step)


def np_slice(matrix, axes={}):
    """Slices a matrix along a specific axes

    Args:
        matrix (numpy.ndarray): the given ndarray martix
        axes (dict): the key present the axis, and the value is a tuple
        representing the slice to make
    Returns:
        (numpy.ndarray): return an ndarray the result of slicing
    Example:
        mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        np_slice(mat1, axes={1: (1, 3)} -> [[2 3]
                                            [7 8]]

    """
    slicers = []
    for i in range(len(matrix.shape)):
        slicers.append(get_slice(axes.get(i)))
    tpl_slicer = tuple(slicers)
    return matrix[tpl_slicer]
