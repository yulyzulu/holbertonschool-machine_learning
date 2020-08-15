#!/usr/bin/env python3
"""Layers"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that create layers"""
    layer = tf.layers.dense(inputs=prev, units=n,
                            activation=activation, name="layer")
    return layer
