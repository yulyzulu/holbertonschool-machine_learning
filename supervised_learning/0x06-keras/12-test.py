#!/usr/bin/env python3
"""Test"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network"""
    test = network.evaluate(x=data, y=labels, verbose=verbose)
    return test
