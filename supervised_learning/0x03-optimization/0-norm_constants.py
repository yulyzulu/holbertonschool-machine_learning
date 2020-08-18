#!/usr/bin/env python3
"""Normalization constants"""

import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization(standarization)
         constants of a matrix"""
#    U = (1/X.shape[0]) * np.sum(X.T)
#    X1 = X - U
#    var2 = 1/X.shape[0] * np.sum(X ** 2)
    U = X.mean(axis=0)
    des = X.std(axis=0)
    return U, des
