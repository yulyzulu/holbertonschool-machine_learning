#!/usr/bin/env python3
""" Convolution with padding"""

import numpy as np
from math import floor, ceil


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a convolution on grayscale images with
        custom padding"""
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    filter_h = kernel.shape[0]
    filter_w = kernel.shape[1]
    pad_h = padding[0]
    pad_w = padding[1]
#    convolve = np.zeros((m, image_h, image_w))
    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant')
    n_dim1 = image_h + 2 * pad_h - filter_h + 1
    n_dim2 = image_w + 2 * pad_w - filter_w + 1
    convolve = np.zeros((m, n_dim1, n_dim2))
    for x in range(n_dim1):
        for y in range(n_dim2):
            mini_matrix = new_images[:, x: x + filter_h, y: y + filter_w]
            values = np.sum(mini_matrix * kernel, axis=1).sum(axis=1)
            convolve[:, x, y] = values
    return (convolve)
