#!/usr/bin/env python3
"""One-hot Encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    try:
        return np.transpose(np.eye(classes)[Y])

    except:
        return None
