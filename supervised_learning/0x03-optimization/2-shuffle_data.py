#!/usr/bin/env python3
"""Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """Function that suffles the data poins in two matrices
       the same way"""
    m = X.shape[0]
    shuff = np.random.permutation(m)
    return X[shuff], Y[shuff]
