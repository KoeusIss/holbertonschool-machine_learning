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
    if not isinstance(poly, list) or poly == []:
        return None
    if len(poly) == 1:
        return [0]
    return [coef * el for coef, el in enumerate(poly) if coef]
