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
                variance = suma / (len(data) - 1)

                self.p = 1 - (variance / mean)
                self.n = int(mean / self.p)
                self.p = float(mean / self.n)
        else:
            if n < 0:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
