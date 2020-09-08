#!/usr/bin/env python3
""" Pooling Forward Prop """

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Function that performs forward propagation over a pooling
         layer of a neural network"""

    m = A_prev.shape[0]
    image_h = A_prev.shape[1]
    image_w = A_prev.shape[2]
    nc = A_prev.shape[3]
    filter_h = kernel_shape[0]
    filter_w = kernel_shape[1]
    s1 = stride[0]
    s2 = stride[1]
    n_dim1 = int((image_h - filter_h) / stride[0]) + 1
    n_dim2 = int((image_w - filter_w) / stride[1]) + 1
    pool = np.zeros((m, n_dim1, n_dim2, nc))
    new_images = A_prev.copy()

    for x in range(n_dim1):
        for y in range(n_dim2):
            mini_matrix = new_images[:, x * s1: x * s1 + filter_h,
                                     y * s2: y * s2 + filter_w, :]
            if mode == 'max':
                values = np.max(mini_matrix, axis=(1, 2))
            else:
                values = np.average(mini_matrix, axis=(1, 2))
            pool[:, x, y, :] = values
    return pool
