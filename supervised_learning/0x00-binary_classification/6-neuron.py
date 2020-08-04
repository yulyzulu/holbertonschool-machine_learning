#!/usr/bin/env python3
"""Neuron class defines a single neuron performing binary classification"""


import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        self.__W = np.random.randn(1, 784)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Method that calculates the forward propagation
            of the neuron"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Method to calculates the cost of the model using
            logistic regression"""
#        print(np.shape(Y))
#        print(len(Y[:, 1]))
#        print(np.size(Y))
        cost = (-1/np.size(Y)) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Method that evaluates the neuron's predictions"""
        prediction = np.zeros((1, np.size(Y)))
        A = self.forward_prop(X)
        for i in range(X.shape[1]):
            prediction[0, i] = 1 if A[0, i] >= 0.5 else 0
            pass
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Method that calculates one pass of gradient
            descent on the neuron"""
        dw = (1/np.size(Y)) * np.matmul(X, (A - Y).T)
        self.__W = self.__W - (alpha * dw)
        db = (1/np.size(Y)) * np.sum(A - Y)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Method that trains the neuron"""
        print(self.nx)
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive") 
