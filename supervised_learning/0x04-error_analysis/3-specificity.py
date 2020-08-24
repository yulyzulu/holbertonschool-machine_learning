#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for each class
    in a confusion matrix"""
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    TN = []
    for i in range(len(confusion)):
        temp = np.delete(confusion, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))

    specificity = TN / (TN + FP)
    return specificity
