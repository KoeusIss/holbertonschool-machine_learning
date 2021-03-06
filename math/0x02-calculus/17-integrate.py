#!/usr/bin/env python3
"""Integrate modulle"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial

    Args:
        poly (list): list of coefficients representing polynomial
        C (int): representing the integration constant
    Return:
        (list|None): list of coefficients representing the integral to the
        polynomial, if poly or c not valid return None

    """
    if not isinstance(poly, list) or poly == []:
        return None
    if not isinstance(C, (int, float)):
        return None
    result = [C]
    for degree, coef in enumerate(poly):
        val = coef * (1 / (degree + 1))
        if val.is_integer():
            result.append(int(val))
        else:
            result.append(val)
    for _ in range(len(result)):
        if result[-1] == 0:
            result.pop()
    return result
