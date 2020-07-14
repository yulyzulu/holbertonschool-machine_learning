#!/usr/bin/env python3
""" Module to execute functions"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise"""
    add_matrix = []
    for i, j in zip(mat1, mat2):
        if len(i) != len(j):
            return None
        else:
            for x, y in zip(i, j):
                add_matrix.append(x + y)
    return add_matrix
