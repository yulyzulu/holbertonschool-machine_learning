#!/usr/bin/env python3
"""Module to execute functions"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polinomial"""
    length = len(poly)
    coefficients = []
    if type(poly) != list or poly == [] or type(C) != int:
        return None
    else:
        coefficients.append(C)
        for i, j in zip(range(1, length + 1), poly):
            integral = (int(j) / int(i)) + C
            coefficients.append(integral)
        return coefficients
