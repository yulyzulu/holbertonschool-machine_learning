#!/usr/bin/env python3
""" Valid convolutional NN"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    m = images.shape[0]
    n_h = images.shape[1]
    n_w = images.shape[2]
    f_h = kernel.shape[0]
    f_w = kernel.shape[1]
    n_dim1 = n_h - f_h + 1
    n_dim2 = n_w - f_w + 1
    convolve = np.zeros((m, n_dim1, n_dim2))
    for x in range(n_dim1):
        for y in range(n_dim2):
            mini_matrix = images[:, x: x + f_h, y: y + f_w]
            values = np.sum(mini_matrix * kernel, axis=1).sum(axis=1)
            convolve[:, x, y] = values
    return (convolve)
