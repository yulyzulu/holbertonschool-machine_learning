#!/usr/bin/env python3
"""L2 Regularization cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """Function that calculates the cost of a neural network
        with L2 regularization"""
    cost_l2 = cost + tf.losses.get_regularization_losses()
    return cost_l2
