#!/usr/bin/env python3
"""Pandas module"""
import pandas as pd
import string


def from_numpy(array):
    """Creates pd.DataFrame from a np.ndarray

    Arguments:
        array {np.ndarray} -- Contains the given array

    Returns:
        pd.DataFrame -- The newly created dataframe.
    """
    _, c = array.shape
    labels = string.ascii_uppercase
    return pd.DataFrame(array, columns=[labels[i] for i in range(c)])
