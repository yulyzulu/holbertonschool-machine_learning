#!/usr/bin/env python3
"""Normalize"""

def normalize(X, m, s):
    """Function that normalizes(standardizes) a matrix"""
    Xnor = (X - m) / s
    return Xnor
