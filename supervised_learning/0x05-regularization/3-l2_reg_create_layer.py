#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes
        L2 Regularization"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lambtha = tf.contrib.layers.l2_regularizer(lambtha)
    l2 = tf.layers.Dense(n, activation, kernel_initializer=W,
                         kernel_regularizer=lambtha)
    return l2(prev)
