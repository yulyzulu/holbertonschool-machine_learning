#!/usr/bin/env python3
"""Layers"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that create layers"""
    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
#                            bias_initializer=tf.zeros_initializer(),
                            name="layer")
    return layer
