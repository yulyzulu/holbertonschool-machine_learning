#!/usr/bin/env python3
"""Inception Block"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(
                             filters=F1,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv1 = conv1(A_prev)
    conv2_1 = K.layers.Conv2D(
                             filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv2_1 = conv2_1(A_prev)
    conv2_2 = K.layers.Conv2D(
                             filters=F3,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv2_2 = conv2_2(conv2_1)
    conv3_1 = K.layers.Conv2D(
                             filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv3_1 = conv3_1(A_prev)
    conv3_2 = K.layers.Conv2D(
                             filters=F5,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv3_2 = conv3_2(conv3_1)
    conv4_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same')
    conv4_pool = conv4_pool(A_prev)
    conv4 = K.layers.Conv2D(
                             filters=FPP,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    conv4 = conv4(conv4_pool)
    output = K.layers.concatenate([conv1, conv2_2, conv3_2, conv4], axis=3)
    return output
