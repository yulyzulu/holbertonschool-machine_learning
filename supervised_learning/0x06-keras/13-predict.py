#!/usr/bin/env python3
"""Predict"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network"""
    predict = network.predict(x=data, verbose=verbose)
    return predict
