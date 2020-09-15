#!/usr/bin/env python3
"""Transition Layer"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer"""
    B_norm1 = K.layers.BatchNormalization()(X)
    act_Relu1 = K.layers.Activation('relu')(B_norm1)
    conv1 = K.layers.Conv2D(int(compression * nb_filters),
                            kernel_size=(1, 1), padding='same',
                            strides=(1, 1),
                            kernel_initializer='he_normal')
    conv1 = conv1(act_Relu1)
    pool_avg = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv1)
    nb_filters = int(nb_filters * compression)
    return pool_avg, nb_filters
