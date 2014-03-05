# Dimensionality Reduction
#
# Goker Erdogan
# 12 Feb 2014

import numpy as np

def pca(x, k):
    """
    Apply PCA on x. Uses np.linalg.svd
    x: input data. matrix of size NxD. N: number of data points. D: input dimensionality
    k: number of output dimensions
    Returns transformed x (Nxk) and projection matrix (Dxk, columns are eigenvectors)
    """
    u, s, v = np.linalg.svd(x, full_matrices=False)
    t = np.dot(u[:,0:k], np.diag(s[0:k]))
    w = v[0:k,:].T 
    return t, w