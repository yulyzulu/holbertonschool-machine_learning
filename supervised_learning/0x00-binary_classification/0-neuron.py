#!/usr/bin/env python3
"""Neuron class defines a single neuron performing binary classification"""


import numpy as np

class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Constructor method"""
        if nx is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        W = np.random.randn()
        b = 0
        A = 0
