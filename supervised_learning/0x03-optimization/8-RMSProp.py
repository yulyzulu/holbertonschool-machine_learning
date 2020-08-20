#!/usr/bin/env python3
"""Update momentum"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function that creates the training operation for a neural network in
       tensorflow using the RMSProp optimization algorithm"""
    Optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    Minim = Optimizer.minimize(loss)
    return Minim
