#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensivity for each class
       in a confusion matrix"""
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP

    sensitivity = TP / (TP + FN)
    return sensitivity
