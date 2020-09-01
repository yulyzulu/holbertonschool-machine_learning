#!/usr/bin/env python3
"""Save and Load Configuration"""

import tensorflow.keras as K


def save_config(network, filename):
    """ Saves a model's configuration in JSON format"""
    json_net = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_net)
    return None


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename) as json_file:
        json_net = json_file.read()
    newNetwork = K.models.model_from_json(json_net)
    return newNetwork
