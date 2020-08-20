#!/usr/bin/env python3
""" Batch Normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of a neural
        network using batch normalization"""
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)

    Znor = (Z - mean) / ((var + epsilon)**(1/2))

    Zn = gamma * Znor + beta
    return Zn
