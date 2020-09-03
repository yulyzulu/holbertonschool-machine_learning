#!/usr/bin/env python3
""" Valid convolutional NN"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    filter_h = kernel.shape[0]
    filter_w = kernel.shape[1]
    pad_h = int((filter_h) / 2)
    pad_w = int((filter_w) / 2)
    convolve = np.zeros((m, image_h, image_w))
    new_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant')
    for x in range(image_h):
        for y in range(image_w):
            mini_matrix = new_images[:, x: x + filter_h, y: y + filter_w]
            values = np.sum(mini_matrix * kernel, axis=1).sum(axis=1)
            convolve[:, x, y] = values
    return (convolve)
