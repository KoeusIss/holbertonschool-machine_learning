#!/usr/bin/env python3
"""Optimization module"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set

    Args:
        data (list): Is the list of data calculating to moving average
        beta (float): Is the weight used for moving average

    Returns:
        list: Containing the moving average

    """
    prev_element = 0
    returned_list = []
    for idx in range(len(data)):
        prev_element = beta * prev_element + (1 - beta) * data[idx]
        bias = 1 - beta**(idx + 1)
        returned_list.append(prev_element / bias)
    return returned_list
