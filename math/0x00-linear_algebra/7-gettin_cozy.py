#!/usr/bin/env python3
"""Module to execute functions"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    m1 = [x[:] for x in mat1]
    m2 = [y[:] for y in mat2]
    concatenate = []

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return m1 + m2

    elif axis == 1 and len(mat1) == len(mat2):
        for i, j in zip(m1, m2):
            concatenate.append(i + j)
        return concatenate
    else:
        return None
