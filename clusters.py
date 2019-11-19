# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:50:22 2019

@author: _Nprime496_
"""


import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=6,cluster_std=0.80, random_state=84)

plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()
input()

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        #print(labels)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');
        plt.show()
        input()
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
        
        
    
    return centers, labels
#input()
centers, labels = find_clusters(X, 4)
#plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');

