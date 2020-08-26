#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network with
        Dropout regularization using gradient descent"""
    m = Y.shape[1]
    num = L
    dZ = cache["A"+str(num)] - Y
    for i in range(L):
        a = str(num)
        b = str(num - 1)
        A = cache["A"+b]
        dW = (1/m) * np.matmul(dZ, A.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        if num - 1 != 0:
            dA = np.matmul(weights["W"+a].T, dZ)
#       if num - 1 != 0:
#        dA = dA * cache["D"+b]
            dZ = dA * (1 - (A**2)) * (cache["D"+b] / keep_prob)
#       dZ = dA / keep_prob
#       dZ = np.multiply(dA, np.int64(A > 0))
        weights["W"+a] = weights["W"+a] - (alpha * dW)
        weights["b"+a] = weights["b"+a] - (alpha * db)
        num = num - 1
