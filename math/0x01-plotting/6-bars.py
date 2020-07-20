#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred','Felicia']
apples = fruit[0:1, :]
bananas = fruit[1:2, :]
oranges = fruit[2:3, :]
peaches = fruit[3:4, :]

x = range(len(names))

bar_width = 0.5
plt.bar(x, apples[0], color='red', width=bar_width, label='apples')
plt.bar(x, bananas[0], bottom=np.array(apples[0]), color='yellow', width=bar_width, label='bananas')
plt.bar(x, oranges[0], bottom=np.array(apples[0])+np.array(bananas[0]), color='#FF8000', width=bar_width, label='oranges')
plt.bar(x, peaches[0], bottom=np.array(apples[0])+np.array(bananas[0])+np.array(oranges[0]), color='#FFE5B4', width=bar_width, label='peaches')
plt.legend()

plt.xticks(x, names)
plt.ylabel('Quantity of Fruit')
plt.ylim([0, 80])
plt.title('Number of Fruit per Person')

plt.show()

