#!/usr/bin/env python3
"""One-Hot Decode"""

import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray or len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    else:
        return np.argmax(one_hot, axis=0)
