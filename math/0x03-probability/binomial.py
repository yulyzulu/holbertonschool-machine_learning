#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor Binomial class"""

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                suma = 0
                for i in data:
                    val = (i - mean) ** 2
                    suma = suma + val
                variance = suma / len(data)

                self.p = 1 - (variance / mean)
                self.n = int(mean / self.p)
                self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)

    def facto(self, n):
        """Method that return the factorial"""
        fac = 1
        for i in range (1, n + 1):
            fac = fac * i
        return fac

    def pmf(self, k):
        """Method that calculates the value of the PMF for a
            given number of successes"""
        if type(k) is not int:
            k = int(k)
        if k > 0:
            return 0
        else:
            C = self.facto(self.n) / ((self.facto(k)) * (self.facto(self.n - k)))
            pmf = C * ((self.p) ** k) * ((1 - self.p)**(self.n - k))
            return pmf
