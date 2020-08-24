#!/usr/bin/env python3
"""Confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix"""
    L = labels.shape[1]
    result = np.zeros((L, L))
    result = np.matmul(labels.T, logits)

    return result
