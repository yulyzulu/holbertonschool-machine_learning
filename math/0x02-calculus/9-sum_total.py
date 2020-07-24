#!/usr/bin/env python3
"""Module to execute functions"""

def summation_i_squared(n):
    """Function that calculates summation"""
    sum = 1
    if type(n) is not int:
        return None
    else:
#        sum = n * ((n + 1)/2)**2
        suma = (n * (n + 1) * ((2 * n) + 1)) / 6
        return suma
