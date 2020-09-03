#!/usr/bin/env python3
""" Convolution with padding"""

import numpy as np
# from math import ceil, floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Function that performs a convolution on grayscale images"""
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    filter_h = kernel.shape[0]
    filter_w = kernel.shape[1]
    s1 = stride[0]
    s2 = stride[1]

    if padding == 'valid':
        pad_h = 0
        pad_w = 0

    if padding == 'same':
        pad_h = int((filter_h - 1) / 2)
        pad_w = int((filter_w -1) / 2)

    if type(padding) == tuple:
        pad_h = padding[0]
        pad_w = padding[1]

    n_dim1 = int((image_h + 2 * pad_h - filter_h + 1) / (stride[0]))
    n_dim2 = int((image_w + 2 * pad_w - filter_w + 1) / (stride[1]))
    convolve = np.zeros((m, n_dim1, n_dim2))
    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant')
    for x in range(n_dim1):
        for y in range(n_dim2):
            mini_matrix = new_images[:, x * s1: x * s1 + filter_h,
                                     y * s2: y * s2 + filter_w]
            values = np.sum(mini_matrix * kernel, axis=1).sum(axis=1)
            convolve[:, x, y] = values
    return (convolve)
