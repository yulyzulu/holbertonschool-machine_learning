#!/usr/bin/env python3
"""Dense Block"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block"""
    for i in range(layers):
        B_norm1 = K.layers.BatchNormalization(axis=3)(X)
        act_Relu1 = K.layers.Activation('relu')(B_norm1)
        conv1 = K.layers.Conv2D(4 * growth_rate, kernel_size=(1, 1),
                                padding='same',
                                strides=(1, 1),
                                kernel_initializer='he_normal')(act_Relu1)

        B_norm2 = K.layers.BatchNormalization(axis=3)(conv1)
        act_Relu2 = K.layers.Activation('relu')(B_norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                                padding='same', strides=(1, 1),
                                kernel_initializer='he_normal')(act_Relu2)
        X = K.layers.Concatenate(axis=3)([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters
