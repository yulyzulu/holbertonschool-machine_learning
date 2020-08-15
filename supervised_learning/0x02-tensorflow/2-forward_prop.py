#!/usr/bin/env python3
"""Forward Propagation"""

import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation
        graph for the Neural Network"""
    create_layer = __import__('1-create_layer').create_layer
    
