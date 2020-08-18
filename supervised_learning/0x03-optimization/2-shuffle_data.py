#!/usr/bin/env python3
"""Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """Function that suffles the data poins in two matrices
       the same way"""
    X_suffle = np.random.permutation(X)
    np.random.seed(0)
    Y_suffle = np.random.permutation(Y)
    return X_suffle, Y_suffle
