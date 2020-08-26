#!/usr/bin/env python3
"""Create Layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Function that """
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    keep_prob = tf.layers.Dropout(keep_prob)
    dropout = tf.layers.Dense(n, activation, kernel_initializer=W,
                              kernel_regularizer=keep_prob)
    return dropout(prev)
