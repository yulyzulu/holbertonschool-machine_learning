#!/usr/bin/env pyhton3
"""Update momentum"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Function that creates the training operation for a
       neural network in tensorflow using the gradient descent
       with momentum optimization algorithm"""
    Optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    Minim = Optimizer.minimize(loss)
    return Minim
