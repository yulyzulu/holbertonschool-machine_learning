#!/usr/bin/env python3
"""Module to execute functions"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polinomial"""
    length = len(poly)
    coefficients = []
    if type(poly) is not list or length == 0:
        return None
    else:
        if length == 1:
            return [0]
        else:
            for i, j in zip(range(0, length), poly):
                if type(j) is not int:
                    return None
                else:
                    der = i * j
                    coefficients.append(der)
        return coefficients[1:]
