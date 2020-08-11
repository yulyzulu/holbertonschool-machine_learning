#!/usr/bin/env python3
"""One-hot Encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    if np.max(Y) > classes or len(Y) == 0:
        return None
    else:
        one = np.transpose(np.eye(classes)[Y])
        return one
