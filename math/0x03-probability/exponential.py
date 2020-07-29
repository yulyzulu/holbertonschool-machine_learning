#!/usr/bin/env python3
"""Exponential distribution"""


class Exponential:
    """Class that represents an exponential distribution"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Constructor class"""
  
        if lambtha < 0:
            raise ValueError('lambtha must be a positive value')
        else:
            self.lambtha = float(lambtha)

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (sum(data)/len(data))
        else:
            self.data = lambtha

    def pdf(self, x):
        """Method that calculates the value of the PDF for
            a given time period"""
        if x < 0:
            return 0
        else:
            d = self.lambtha * (self.e ** (-self.lambtha * x))
            return d

    def cdf(self, x):
        """Method that calculates the value of the CDF for
            a given time period"""
        if x < 0:
            return 0
        else:
            c = 1 - (self.e ** (-self.lambtha * x))
            return c
