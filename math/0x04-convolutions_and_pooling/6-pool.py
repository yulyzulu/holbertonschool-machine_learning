#!/usr/bin/env python3
""" Convolution with padding"""

import numpy as np
from math import floor, ceil


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images"""

    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    nc = images.shape[3]
    filter_h = kernel_shape[0]
    filter_w = kernel_shape[1]
    s1 = stride[0]
    s2 = stride[1]
    n_dim1 = int(floor(float(image_h - filter_h + 1) / float(stride[0])))
    n_dim2 = int(floor(float(image_w - filter_w + 1) / float(stride[1])))
    pool = np.zeros((m, n_dim1, n_dim2, nc))
    new_images = images.copy()
#    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
#                        mode='constant')
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
