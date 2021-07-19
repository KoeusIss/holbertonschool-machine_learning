#!/usr/bin/env python3
"""Pandas module"""
import pandas as pd
import numpy as np


def from_file(filename, delimiter):
    """Reads data from file.

    Arguments:
        filename {string} -- Is the filename
        delimiter {string} -- Is the delimiter used for speration

    Returns:
        pd.DataFrame -- A newly created dataframe
    """
    return pd.read_csv(filename, delimiter=delimiter)
