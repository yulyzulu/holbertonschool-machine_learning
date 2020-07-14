#!/usr/bin/env python3
"""Module to execute functions"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix"""
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
        for j in range(len(matrix)):
            transpose[i].append(matrix[j][i])
    return transpose
