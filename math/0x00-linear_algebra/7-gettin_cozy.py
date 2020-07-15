#!/usr/bin/env python3
"""Module to execute functions"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        for i in mat1:
            for j in mat2:
                return i + j
    else:
        return None
