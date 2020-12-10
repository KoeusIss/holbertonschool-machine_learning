#!/usr/bin/env python3
"""Derivatives modules"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial

    Args:
        poly (list): the list of coeficient representing a polynomial

    Returns:
        (list|None): returns a new list of coeficient representing
        the derivative of the polynomial, if poly is not valid return None.

    """
    if not isinstance(poly, list) or not all(
            [isinstance(x, (int, float)) for x in poly]
    ):
        return None
    result = [coef * el for coef, el in enumerate(poly) if coef]
    if len(poly) == 1 or not any(result):
        return [0]
    else:
        return result
