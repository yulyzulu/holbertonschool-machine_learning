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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        n_layer = 1
        n_node = nx

        for n in layers:
            if type(n) is not int or n < 0:
                raise TypeError("layers must be a list of positive integers")
            W = "W" + str(n_layer)
            b = "b" + str(n_layer)
            self.__weights[W] = np.random.randn(n, n_node)*np.sqrt(2/n_node)
            self.__weights[b] = np.zeros((n, 1))
            n_layer = n_layer + 1
            n_node = n

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Method that calculates the forward propagation of the neural
            network"""
        self.__cache["A0"] = X
        n_layer = 1
        for i in range(self.__L):
            a = str(n_layer)
            b = str(n_layer - 1)
            WX = np.dot(self.__weights["W"+a], self.__cache["A"+b])
            Z = WX + self.__weights["b"+a]
            self.__cache["A"+str(n_layer)] = 1 / (1 + np.exp(-Z))
            n_layer = n_layer + 1
        return self.__cache["A"+str(n_layer-1)], self.__cache

    def cost(self, Y, A):
        """Method that calculates the cost of the model using logistic
           regression"""
        loss = -((Y * np.log(A)) + (1 - Y) * (np.log(1.0000001 - A)))
        cost = (1/np.size(Y)) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Method that evaluates the neural network´s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.round(A).astype(int)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Method that calculates one pass of gradient descent on the
           neural network"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache["A"+str(L)] - Y
        for i in range(self.__L):
            a = str(L)
            b = str(L - 1)
            A = cache["A"+b]
            dW = (1/m) * np.matmul(dZ, A.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            self.__weights["W"+a] = self.__weights["W"+a] - (alpha * dW)
            self.__weights["b"+a] = self.__weights["b"+a] - (alpha * db)
#            A = cache["A"+b]
            dZ = np.matmul(self.__weights["W"+a].T, dZ) * (A * (1-A))
            L = L - 1
