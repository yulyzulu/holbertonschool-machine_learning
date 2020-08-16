#!/usr/bin/env python3
"""Accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a preddiction"""
#    accuracy = tf.sum(y == y_pred) / y.shape[0] * 100
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy
