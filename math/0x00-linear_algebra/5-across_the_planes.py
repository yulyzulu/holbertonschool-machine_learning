#!/usr/bin/env python3
""" Module to execute functions"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        add_matrix = []
        for i in range(len(mat1)):
            add_matrix.append([])
            for j in range(len(mat1[0])):
                add_matrix[i].append(mat1[i][j] + mat2[i][j])
    return add_matrix
