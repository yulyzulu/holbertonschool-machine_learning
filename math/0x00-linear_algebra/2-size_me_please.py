#!/usr/bin/env python3
""" Module to execute function """


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    shape = []
    shape.append(len(matrix))
    while type(matrix[0]) == list:
        matrix = matrix[0]
        shape.append(len(matrix))
    return shape
