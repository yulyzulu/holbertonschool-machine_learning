#!/usr/bin/env python3
"""Batch normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a
         neural network in tensorflow"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=W,
                            name="layer")
    X = layer(prev)
    mean, variance = tf.nn.moments(X, [0])
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    X_norm = tf.nn.batch_normalization(X, mean, variance, offset=beta,
                                       scale=gamma, variance_epsilon=1e-8)
    return activation(X_norm)
