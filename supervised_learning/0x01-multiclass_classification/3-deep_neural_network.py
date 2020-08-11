#!/usr/bin/env python3
"""Deep Neural Network"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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
            if i != self.__L - 1:
                self.__cache["A"+str(n_layer)] = 1 / (1 + np.exp(-Z))
#            n_layer = n_layer + 1
#        WX = np.dot(self.__weights["W"+str(n_layer)], self.__cache["A"+str(n_layer-1)])
#        Z = WX + self.__weights["b"+str(n_layer)]
            else:
                expZ = np.exp(Z)
                self.__cache["A"+str(n_layer)] = expZ / expZ.sum(axis=0, keepdims=True)
            n_layer = n_layer + 1
        return self.__cache["A"+str(n_layer-1)], self.__cache

    def cost(self, Y, A):
        """Method that calculates the cost of the model using logistic
           regression"""
        m = Y.shape[1]
        loss = np.log(A) * Y
        cost = -(1/m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Method that evaluates the neural network´s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        Ad = np.amax(A, axis=0)
        prediction = np.where(A == Ad, 1, 0)
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
            dZ = np.matmul(self.__weights["W"+a].T, dZ) * (A * (1-A))
            self.__weights["W"+a] = self.__weights["W"+a] - (alpha * dW)
            self.__weights["b"+a] = self.__weights["b"+a] - (alpha * db)
            L = L - 1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Method that trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        iteration = []
        costs = []
        for i in range(0, iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i is step or i % step is 0:
                if verbose is True:
                    print("Cost after", i, "iterations:", cost)
                    iteration.append(i)
                    costs.append(cost)

            if i != iterations:
                self.gradient_descent(Y, cache, alpha)
                prediction, cost = self.evaluate(X, Y)

        if graph is True:
            plt.plot(iteration, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
#       prediction, cost = self.evaluate(X, Y)
        return prediction, cost

    def save(self, filename):
        """Method that saves the instance object to a file"""
        x = filename.split(".")
        if x[-1] != "pkl":
            filename = filename + ".pkl"

        with open(filename, 'wb') as fileObject:
            pickle.dump(self, fileObject)


    def load(filename):
        """Static method that loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as fileObject:
                obj = pickle.load(fileObject)
                return obj
        except (OSError, IOError) as e:
            return None
