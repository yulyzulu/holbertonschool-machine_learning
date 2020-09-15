#!/usr/bin/env python3
"""Dense Block"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block"""
