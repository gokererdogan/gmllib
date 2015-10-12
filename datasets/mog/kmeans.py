"""
gmllib

Applies kmeans to mog dataset.

Goker Erdogan
2015
https://github.com/gokererdogan
"""
import numpy as np
import gmllib.dataset as dataset
import gmllib.clustering as clustering

if __name__ == "__main__":
    ds = dataset.DataSet(trainx='mog_trainx.npy', trainy='mog_trainy.npy',
                         testx='mog_testx.npy', testy='mog_testy.npy')

    ground_truth = np.array([[-1.5, -1], [1, 1], [-1, 2]])
    print("Ground truth:\n {0:s}".format(ground_truth))

    mh = clustering.k_means(dataset=ds, K=3, method='hard')
    print("\nHard k-means:\n {0:s}".format(mh))

    ms = clustering.k_means(dataset=ds, K=3, method='soft', beta=5.0)
    print("\nSoft k-means:\n {0:s}".format(ms))

