#!/usr/bin/env python3
""" Poisson distribution"""


class Poisson:
    """Class Poisson"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Class Poisson that represents a poisson distribution"""

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = sum(data)/len(data)
        else:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)

    def facto(self, n):
        """Function that return the factorial"""
        fac = 1
        for i in range(1, n + 1):
            fac = fac * i
        return fac

    def pmf(self, k):
        """Method to calculates the value of the PMF por a fiven
            number of successes"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            p = self.e ** -self.lambtha * (self.lambtha ** k) / self.factor(k)
            return p

    def cdf(self, k):
        """Method that calculates the value of the CDF for a given
            number of succeses"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            suma = 0
            for i in range(0, k + 1):
                p = self.e ** -self.lambtha * self.lambtha ** i / self.facto(i)
                suma = suma + p
            return suma
