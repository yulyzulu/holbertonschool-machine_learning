#!/usr/bin/env python3
"""Module to execute functions"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polinomial"""
    length = len(poly)
    coefficients = []
    if type(poly) != list or length == 0:
        return None
    else:
        if length == 1:
            return [0]
        else:
            for i in range(1, length):
                if type(poly[i]) != int and type(poly[i]) != float:
                    return None
                else:
                    der = i * poly[i]
                    coefficients.append(der)
        return coefficients
