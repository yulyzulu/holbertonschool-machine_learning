#!/usr/bin/env python3
"""Prediction"""
import numpy as np


def precision(confusion):
    """Function that calculates the precision for each class
    in a confusion matrix"""
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    precision = TP / (TP + FP)
    return precision
