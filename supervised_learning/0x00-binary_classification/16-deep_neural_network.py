#!/usr/bin/env python3
"""Deep Neural Network that defines a deep neural network performing
    binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network Class"""
    def __init__(self, nx, layers):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        n_layer = 1
        n_node = nx

        for n in layers:
            if type(n) is not int or n < 0:
                raise ValueError("layers must be a list of positive integers")
            W = "W" + str(n_layer)
            b = "b" + str(n_layer)
            self.weights[W] = np.random.randn(n, n_node)*np.sqrt(2/n_node)
            self.weights[b] = np.zeros((n, 1))
            n_layer = n_layer + 1
            n_node = n
