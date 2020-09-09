#!/usr/bin/env python3
""" Pooling Back Prop """

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation over a pooling
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
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for x in range(n_dim1):
            for y in range(n_dim2):
                for c in range(nc):
                    mini_matrix = A_prev[i, x * s1: x * s1 + filter_h,
                                         y * s2: y * s2 + filter_w, c]
                    if mode == 'max':
                        mask = (mini_matrix.max() == mini_matrix)
                        dA_prev[i, x * s1: x * s1 + filter_h,
                                y * s2: y * s2 + filter_w,
                                c] += dA[i, x, y, c] * mask
                    else:
                        dA_prev[i, x * s1: x * s1 + filter_h,
                                y * s2: y * s2 + filter_w,
                                c] += dA[i, x, y, c] / (filter_h * filter_w)
    return dA_prev
