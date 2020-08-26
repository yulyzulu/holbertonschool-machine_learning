#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using
         Dropout"""
    cache = {}
    cache["A0"] = X
    n_layer = 1
    for i in range(L):
        a = str(n_layer)
        b = str(n_layer - 1)
        WX = np.dot(weights["W"+a], cache["A"+b])
        Z = WX + weights["b"+a]
        if i != L - 1:
            numer = np.exp(Z) - np.exp(-Z)
            denom = np.exp(Z) + np.exp(-Z)
            A = numer / denom
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = A * D
            A = A / keep_prob
            cache["A"+a] = A
            cache["D"+a] = D
        else:
            expZ = np.exp(Z)
            A = expZ / expZ.sum(axis=0, keepdims=True)
            cache["A"+a] = A
        n_layer = n_layer + 1
    return cache
