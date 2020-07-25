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
        for num in reversed(poly):
            if num == 0:
                poly.pop(num)
            else:
                break
        for i, j in zip(range(1, length + 1), poly):
            integral = j / i
            if int(integral) == integral:
                coefficients.append(int(integral))
            else:
                coefficients.append(integral)
        return coefficients