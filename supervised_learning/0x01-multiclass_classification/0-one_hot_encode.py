#!/usr/bin/env python3
"""One-hot Encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    try:
        one = np.transpose(np.eye(classes)[Y])
        return one

    except Exception:
        return None
