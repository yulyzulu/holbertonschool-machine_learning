#!/usr/bin/env python3
"""Loss"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Function that calculates the softmax cross-entropy
        loss of a predicction"""
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
