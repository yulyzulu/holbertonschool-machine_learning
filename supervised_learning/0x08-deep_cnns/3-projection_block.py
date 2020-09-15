#!/usr/bin/env python3
""" Projection Block"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block"""
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(
                             filters=F11,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding='same',
                             kernel_initializer='he_normal')
    conv1 = conv1(A_prev)
    B_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act_Relu1 = K.layers.Activation('relu')(B_norm1)
    conv2 = K.layers.Conv2D(
                             filters=F3,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='same',
                             kernel_initializer='he_normal')
    conv2 = conv2(act_Relu1)
    B_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    act_Relu2 = K.layers.Activation('relu')(B_norm2)
    conv3 = K.layers.Conv2D(
                             filters=F12,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding='same',
                             kernel_initializer='he_normal')
    conv3 = conv3(act_Relu2)
    B_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    X_shortcut = K.layers.Conv2D(
                             filters=F12,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding='same',
                             kernel_initializer='he_normal')
    X_shortcut = X_shortcut(A_prev)
    B_norm4 = K.layers.BatchNormalization(axis=3)(X_shortcut)

    add_l = K.layers.Add()([B_norm3, B_norm4])
    act_Relu3 = K.layers.Activation('relu')(add_l)
    return act_Relu3
