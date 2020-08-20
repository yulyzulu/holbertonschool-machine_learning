#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place using the Adam
       optimization algorithm"""

    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    Sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    Vdw_co = Vdw / (1 - beta1 ** t)
    Sdw_co = Sdw / (1 - beta2 ** t)
    W = var - (alpha * (Vdw_co / (Sdw_co ** (1/2) + epsilon)))

    return W, Vdw, Sdw
