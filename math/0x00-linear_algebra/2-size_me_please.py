#!/usr/bin/env python3
""" Module to execute function """


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    rows = 0
    columns = 0
    count = 0
    shape = []
    for i in matrix:
        rows = rows + 1
        for j in i:
            columns = len(i)
            if type(j) == list:
                for k in j:
                    count = len(j)
    shape.append(rows)
    shape.append(columns)
    if count > 0:
        shape.append(count)
    return shape
