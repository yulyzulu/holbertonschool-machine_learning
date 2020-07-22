#!/usr/bin/env python3
"""Print a plot """
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(0, 10, y, 'r-')
plt.xlim([0, 10])
plt.show()
