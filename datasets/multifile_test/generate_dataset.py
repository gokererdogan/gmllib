"""
gmllib

This scripts generate a dataset for testing MultiFileArray class.

Goker Erdogan
2015
https://github.com/gokererdogan
"""


import numpy as np

x = np.arange(0, 200)
x1 = np.ones((100, 2))
x2 = np.ones((60, 2)) * 2
x3 = np.ones((40, 2)) * 3
x1[:, 1] = x[0:100]
x2[:, 1] = x[100:160]
x3[:, 1] = x[160:200]

y1 = np.ones(100)
y2 = np.ones(60) * 2
y3 = np.ones(40) * 3

np.save('mf_trainx1.npy', x1)
np.save('mf_trainx2.npy', x2)
np.save('mf_trainx3.npy', x3)

np.save('mf_trainy1.npy', y1)
np.save('mf_trainy2.npy', y2)
np.save('mf_trainy3.npy', y3)
