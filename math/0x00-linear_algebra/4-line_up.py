#!/usr/bin/env python3
""" Module to execute functions"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    add = []
    if len(arr1) is not len(arr2):
        return None
    else:
        for i, j in zip(arr1, arr2):
            add.append(i + j)
    return add
