#!/usr/bin/env python3
""" Module to execute function """


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    shape = []
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    i = matrix[0]
    if type(i[0]) == list:
        shape.append(len(i[0]))
    return shape
