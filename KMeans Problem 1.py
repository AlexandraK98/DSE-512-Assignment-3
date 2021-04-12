#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load Data 
import requests 
link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
f = requests.get(link)

import pandas as pd

# the directory contains a labels.csv which we will not need for clustering
features = pd.read_csv('/Users/ack98/Downloads/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0)
print(features.shape)
print(features.head())

from sklearn.cluster import KMeans
kmeans = KMeans (n_clusters=4)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='gene_0')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[ ]:


#Run Kmeans

import numpy as np 
def kmeans(xs, num_clusters=4):
    """Run k-means algorithm to convergence.

    Args:
        xs: numpy.ndarray: An N-by-d array describing N data points each of dimension d
        num_clusters: int: The number of clusters desired
    """
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space
    # INITIALIZATION PHASE
    # initialize centroids randomly as distinct elements of xs
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids  = xs[cids, :]
    assignments = np.zeros(N, dtype=np.uint8)

    # loop until convergence
    import time
    start = time.perf_counter()
    it = 0
    while True:
        it += 1
        # Compute distances from sample points to centroids
        # all  pair-wise _squared_ distances
        cdists = np.zeros((N, num_clusters))
        for i in range(N):
            xi = xs[i, :]
            for c in range(num_clusters):
                cc  = centroids[c, :]
                dist = 0
                for j in range(d):
                    dist += (xi[j] - cc[j]) ** 2
                cdists[i, c] = dist
        # Expectation step: assign clusters
        num_changed_assignments = 0
        for i in range(N):
            # pick closest cluster
            cmin = 0
            mindist = np.inf
            for c in range(num_clusters):
                if cdists[i, c] < mindist:
                    cmin = c
                    mindist = cdists[i, c]
            if assignments[i] != cmin:
                num_changed_assignments += 1
            assignments[i] = cmin

        # Maximization step: Update centroid for each cluster
        for c in range(num_clusters):
            newcent = 0
            clustersize = 0
            for i in range(N):
                if assignments[i] == c:
                    newcent = newcent + xs[i, :]
                    clustersize += 1
            newcent = newcent / clustersize
            centroids[c, :]  = newcent

        print(f"Iteration {it}. Elapsed time: {time.perf_counter() - start} seconds", flush=True)

        if num_changed_assignments == 0:
            break

    endtime = time.perf_counter()
    print(f"Converged in {it} iterations in {endtime - start} seconds")

    # return cluster centroids and assignments
    return centroids, assignments


'''if __name__ == '__main__':
    # take arguments like number of clusters k
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', choices=('tcga')
    parser.add_argument('-k', type = int, required=True, help="Number of clusters")
    args = parser.parse_args()

    if args.dataset == 'iris':
        from sklearn.datasets import load_iris
        features, labels = load_iris(return_X_y=True)
    elif args.dataset == 'tcga':'''
import pandas as pd
features = pd.read_csv('/Users/ack98/Downloads/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0).to_numpy()


# run k-means
centroids, assignments = kmeans(features, num_clusters=4)  

# print out results
print(centroids, assignments)

