#!/usr/bin/env python3
"""Module to execute functions"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polinomial"""
    length = len(poly)
    coefficients = []
    if type(poly) != list:
        return None
    else:
        if length == 1:
            return [0]
        else:
            for i, j in zip(range(length), poly):
                der = i * j
# if j != poly[0]:
                coefficients.append(der)
        return coefficients[1:]
