#!/usr/bin/env python3
"""Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """Function that suffles the data poins in two matrices
       the same way"""
    X_suffle = X[np.random.permutation(X.shape[0])]
    np.random.seed(0)
    Y_suffle = Y[np.random.permutation(X.shape[0])]
    return X_suffle, Y_suffle
