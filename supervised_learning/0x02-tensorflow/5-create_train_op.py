#!/usr/bin/env python3
"""Train_Op"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the operation for the network"""
    gra = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return gra
