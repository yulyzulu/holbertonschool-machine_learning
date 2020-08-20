#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """Function that calculates the weighted moving average
       of a data set"""
    V_prev = 0
    W_Average = []
    for i in range(len(data)):
        V = (beta * V_prev) + (1 - beta) * data[i]
        C_bias = (1 - beta ** (i + 1))
        AV = V / C_bias
        W_Average.append(AV)
        V_prev = V

    return W_Average
