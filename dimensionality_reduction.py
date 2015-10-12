# Dimensionality Reduction
#
# Goker Erdogan
# 12 Feb 2014

import numpy as np
import gmllib.dataset as ds

def pca(dataset, k):
    """
    Apply PCA on x. Uses np.linalg.svd
    x: input data. DataSet instance or matrix of size NxD.
        N: number of data points. D: input dimensionality
    k: number of output dimensions
    Returns transformed x (Nxk), explained variance by component
    and projection matrix (Dxk, columns are eigenvectors)
    """
    if isinstance(dataset, ds.DataSet):
        x = dataset.train.x
    else:
        x = dataset

    u, s, v = np.linalg.svd(x, full_matrices=False)
    z = np.dot(u[:, 0:k], np.diag(s[0:k]))
    var = np.square(s) / np.sum(np.square(s))
    var = var[0:k]
    w = v[0:k, :].T
    return z, var, w
