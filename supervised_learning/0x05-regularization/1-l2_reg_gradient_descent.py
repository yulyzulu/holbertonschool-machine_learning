#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network
       using gradient descent with L2 regularization"""
    m = Y.shape[1]
    dZ = cache["A"+str(L)] - Y
    for i in range(L):
        a = str(L)
        b = str(L-1)
        A = cache["A"+b]
        dW = (1/m) * np.matmul(dZ, A.T) + (lambtha/m * weights["W"+a])
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.matmul(weights["W"+a].T, dZ) * (A * (1-A))
        weights["W"+a] = weights["W"+a] - (alpha * dW)
        weights["b"+a] = weights["b"+a] - (alpha * db)
        L = L - 1
