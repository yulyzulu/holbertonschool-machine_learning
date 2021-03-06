#!/usr/bin/env python3
"""Neural Network class defines a neural network with one
    hidden layer performing binary classification"""


import numpy as np


class NeuralNetwork:
    """NeuralNetwork Class"""
    def __init__(self, nx, nodes):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.nodes = nodes

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Method that calculates the forward propagation of the
            neural network"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Method to calculates the cost of the model using
           logistic regression"""
        loss = -((Y * np.log(A)) + (1 - Y) * (np.log(1.0000001 - A)))
        cost = (1/np.size(Y)) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Method that evaluates the neural network predictions"""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.round(A2).astype(int)
        return prediction, cost
