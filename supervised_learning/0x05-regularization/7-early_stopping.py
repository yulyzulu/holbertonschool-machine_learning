#!/usr/bin/env python3
"""Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you should stop gradient descent early"""
    dif = opt_cost - cost
    val = True
    if dif > threshold:
        count = 0
    else:
        count = count + 1
    if count != patience:
        val = False
    return val, count
