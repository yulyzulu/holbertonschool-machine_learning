#!/usr/bin/env python3
"""Module to execute functions"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    concatenate = []
    if axis == 0:
        return mat1 + mat2 
    elif axis == 1:
        for i, j in zip(mat1, mat2):
            concatenate.append(i + j)
            return concatenate
    else:
        return None
