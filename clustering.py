# Clustering functions
# Goker Erdogan
# 29 Jan 2013

import numpy as np
import scipy.spatial.distance as dist
import dataset as ds

def k_means(dataset, K, method='hard', distance_metric='euclidean', stop_tol=1e-9, beta=None):
    """
    K-means clustering with hard and soft constraints.
    dataset: A Dataset instance containing training data or
        a numpy array (data points in rows, attributes in columns)
    K: number of clusters
    method: 'soft' or 'hard'
    distance_metric: any of the distance metrics accepted by scipy.spatial.distance.cdist
    stop_tol: stop if change in means is less than stop_tol
    beta: stiffness parameter for soft k-means
    Returns:
        m: matrix cluster means (as rows of the matrix)
    Ref: Mackay, D. Information Theory, Inference, and Learning Algorithms. Ch. 20
    """
    # check parameters
    if isinstance(dataset, ds.DataSet):
        x = dataset.train.x
        D = dataset.D
        N = dataset.train.N
    else:
        x = dataset
        D = np.size(x, 1)
        N = np.size(x, 0)

    if method not in ('soft', 'hard'):
        raise Exception('k-means: method should be \'soft\' or \'hard\'.')
    if method == 'soft' and not beta:
        raise Exception('k-means: beta parameter is required for soft k-means.')
    
    # initialize means randomly from range of x
    m = ((np.random.rand(K, D) - 0.5) * 2) * np.max(x, axis=0)
    
    diff_m = 1
    while diff_m > stop_tol:
        m_old = m
        
        # calculate pairwise distances
        d = dist.cdist(x, m, metric=distance_metric)
        # calculate responsibilities
        if method == 'hard':
            r = (d == np.min(d, axis=1, keepdims=True)).astype(int)
        else: # soft k-means
            r = np.exp(-d*beta) / np.sum(np.exp(-d*beta), axis=1, keepdims=1)
        
        # update m
        # beware of clusters with no assigned points (r=0)
        tot_r = np.sum(r, axis=0).T
        sum_rx = np.dot(r.T, x)
        nonempty_clusters = (tot_r != 0)
        m[nonempty_clusters] = sum_rx[nonempty_clusters] / np.tile(tot_r[nonempty_clusters], (D, 1)).T
        
        # calculate change in m
        diff_m = np.sum((m - m_old)**2)
        
    return m
