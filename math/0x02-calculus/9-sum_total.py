#!/usr/bin/env python3
"""Module to execute functions"""


def summation_i_squared(n):
    """Function that calculates summation"""
    if type(n) is not int:
        return None
    else:
        summation = (n * (n + 1) * ((2 * n) + 1)) / 6
        return int(summation)
