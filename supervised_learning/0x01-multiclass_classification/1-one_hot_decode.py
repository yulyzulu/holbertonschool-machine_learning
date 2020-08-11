#!/usr/bin/env python3
"""One-Hot Decode"""

import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels"""
    try:
        return np.argmax(one_hot, axis=0)
    except:
        return None
