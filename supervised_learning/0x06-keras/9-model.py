#!/usr/bin/env python3
"""Save and load model"""

import tensorflow.keras as K


def save_model(network, filename):
    """ Save model"""
    network.save(filename)
    return None


def load_model(filename):
    """Load model"""
    load = K.models.load_model(filename)
    return load
