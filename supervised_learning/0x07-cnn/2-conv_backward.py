#!/usr/bin/env python3
""" Convolutional Back Prop"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional
        layer of a neural network """
    m = A_prev.shape[0]
    image_h = A_prev.shape[1]
    image_w = A_prev.shape[2]
    filter_h = W.shape[0]
    filter_w = W.shape[1]
    c = W.shape[2]
    nc = W.shape[3]
    s1 = stride[0]
    s2 = stride[1]

    if padding == 'valid':
        pad_h = 0
        pad_w = 0

    if padding == 'same':
        pad_h = int(((image_h - 1) * s1 + filter_h - image_h) / 2) + 1
        pad_w = int(((image_w - 1) * s2 + filter_w - image_w) / 2) + 1

    n_dim1 = int((image_h + 2 * pad_h - filter_h) / stride[0]) + 1
    n_dim2 = int((image_w + 2 * pad_w - filter_w) / stride[1]) + 1
#    convolve = np.zeros((m, n_dim1, n_dim2, nc))
    new_image = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                (0, 0)), mode='constant')
    dA_prev = np.zeros(new_image.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for n in range(m):
        for x in range(n_dim1):
            for y in range(n_dim2):
                for f in range(nc):
                    dA_prev[n, x * s1: x * s1 + filter_h,
                            y * s2: y * s2 + filter_w,
                            :] += dZ[n, x, y, f] * W[:, :, :, f]
                    dW[:, :, :, f] += new_image[n, x * s1: x * s1 + filter_h,
                                                y * s2: y * s2 + filter_w,
                                                :] * dZ[n, x, y, f]
    if padding == 'same':
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA_prev, dW, db
