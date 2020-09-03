#!/usr/bin/env python3
""" Convolution with Channels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that that performs a convolution on images
         using multiple kernels"""
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    filter_h = kernels.shape[0]
    filter_w = kernels.shape[1]
    c = kernels.shape[2]
    nc = kernels.shape[3]
    s1 = stride[0]
    s2 = stride[1]

    if padding == 'valid':
        pad_h = 0
        pad_w = 0

    if padding == 'same':
        pad_h = int(((image_h - 1) * s1 + filter_h - image_h) / 2) + 1
        pad_w = int(((image_w - 1) * s2 + filter_w - image_w) / 2) + 1

    if type(padding) == tuple:
        pad_h = padding[0]
        pad_w = padding[1]

    n_dim1 = int((image_h + 2 * pad_h - filter_h) / stride[0]) + 1
    n_dim2 = int((image_w + 2 * pad_w - filter_w) / stride[1]) + 1
    convolve = np.zeros((m, n_dim1, n_dim2, nc))
    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                 (0, 0)), mode='constant')
    for d in range(nc):
        for x in range(n_dim1):
            for y in range(n_dim2):
                mini_matrix = new_images[:, x * s1: x * s1 + filter_h,
                                         y * s2: y * s2 + filter_w, :]
                values = np.sum(mini_matrix * kernels[..., d],
                                axis=1).sum(axis=1).sum(axis=1)
                convolve[:, x, y, d] = values
    return (convolve)
