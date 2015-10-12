"""
gmllib

Generates a simple mixture of Gaussians of 3 clusters
This dataset is intended for testing clustering algorithms.

Goker Erdogan
2015
https://github.com/gokererdogan
"""
import numpy as np
import scipy.stats as stat
import gmllib.helpers as hlp

samples_per_cluster = 150

x1 = stat.multivariate_normal.rvs(mean=[-1.5, -1.], cov=np.diag([.5, .25]), size=samples_per_cluster)
x2 = stat.multivariate_normal.rvs(mean=[1., 1.], cov=np.diag([.1, .3]), size=samples_per_cluster)
x3 = stat.multivariate_normal.rvs(mean=[-1, 2], cov=np.diag([.1, .1]), size=samples_per_cluster)

tx1 = x1[0:100, :]
tx2 = x2[0:100, :]
tx3 = x3[0:100, :]
tx = np.vstack((tx1, tx2, tx3))
ty = np.ones(300)
ty[100:200] = 2
ty[200:300] = 3
tx, ty = hlp.shuffle_dataset(tx, ty)

sx1 = x1[100:150, :]
sx2 = x2[100:150, :]
sx3 = x3[100:150, :]
sx = np.vstack((sx1, sx2, sx3))
sy = np.ones(150)
sy[50:100] = 2
sy[100:150] = 3
sx, sy = hlp.shuffle_dataset(sx, sy)

np.save('mog_trainx.npy', tx)
np.save('mog_trainy.npy', ty)
np.save('mog_testx.npy', sx)
np.save('mog_testy.npy', sy)
