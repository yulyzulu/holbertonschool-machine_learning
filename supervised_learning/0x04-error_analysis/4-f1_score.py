#!/usr/bin/env python3
"""F1 score"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function that calcualtes the F1 score of a confusion matrix"""
    precision1 = precision(confusion)
    sensitivity1 = sensitivity(confusion)
    F1_score = (2 * precision1 * sensitivity1 / (precision1 + sensitivity1))
    return F1_score
