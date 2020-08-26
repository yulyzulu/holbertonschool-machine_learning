#!/usr/bin/env python3
"""L2 regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network
        with L2 regularization"""
    a = 1
    regl2 = 0
    for i in range(L):
        w = "W"+str(a)
        regl2 = regl2 + np.linalg.norm(weights[w], None)
        a = a + 1
    new_cost = cost + (lambtha/(2*m)) * (regl2)
    return new_cost
