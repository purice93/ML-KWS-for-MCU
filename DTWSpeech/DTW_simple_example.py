""" 
@author: zoutai
@file: DTW_simple_example.py 
@time: 2018/12/12 
@description: 
"""

import matplotlib.pyplot as plt
import numpy as np

from DTWSpeech.dtw import dtw

x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

plt.plot(x)
plt.plot(y)
# plt.show()

dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
print('Minimum distance found:', dist)

'''
### You can plot the accumulated cost matrix and the "shortest" wrap path.
plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))
plt.show()
'''


def my_custom_norm(x, y):
    return (x * x) + (y * y)


x = range(10)
y = np.append([0] * 5, x)

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

dist, cost, acc, path = dtw(x, y, dist=my_custom_norm)
plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0] - 0.5))
plt.ylim((-0.5, acc.shape[1] - 0.5))
plt.show()
