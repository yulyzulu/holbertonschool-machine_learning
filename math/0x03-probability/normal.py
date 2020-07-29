#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Normal class"""
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor"""
        if stddev < 0:
            raise ValueError("stddev must be a positive value")
        else:
            self.stddev = float(stddev)

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            suma = 0
            for i in data:
                dif = (i - self.mean) ** 2
                suma = suma + dif
            self.stddev = float((suma / len(data)) ** (1/2))

        else:
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """Method that calculates de z-score of a given x-value"""
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """Method that calculates the x-value of a given z-score"""
        x_v = (self.stddev * z) + self.mean
        return x_v

    def pdf(self, x):
        """Method that calculates the value of the PDF for
            a given x-value"""
        p =  1 / (self.stddev * ((2 * self.pi) ** (1/2))) * (self.e ** -(((x - self.mean) ** 2) / 2 * (self.stddev ** 2)))
        return p
