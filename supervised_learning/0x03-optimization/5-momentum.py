#!/usr/bin/env python3
"""Momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the gradient
       descent with momentum optimization algorithm"""
    Vdw = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * Vdw)
    return W, Vdw
