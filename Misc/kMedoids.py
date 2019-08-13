#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:22:17 2019

@author: dileepn

k-Medoids: A Toy Example 
"""
import numpy as np

# Data points as a matrix
X = np.array([[0,-6],[4,4],[0,0],[-5,2]])

# Initialize representatives
z = np.array([[-5,2],[0,-6]])

# Cluster affiliations
n = X.shape[0]
k = z.shape[0]
clusters = np.zeros((n, 1))

# No. of loops through the dataset
loops = 10

cost_old = 0.0

for l in range(loops):
    # Assign points to clusters
    cost_new = 0.0
    for i in range(n):
        dists = [0.0, 0.0]
        for j in range(k):
            dists[j] = np.linalg.norm(X[i,:] - z[j,:], ord = 1)
            clusters[i] = np.argmin(dists)
        
        cost_new = cost_new + min(dists)    # Compute cost
        
    # If cost doesn't change, end
    if np.abs(cost_new - cost_old) < 1e-8:
        print("Number of loops through the dataset:", l+1)
        break
    else:
        cost_old = cost_new

    # Select new representatives 
    dists = np.zeros((k,k))
    for j in range(n):
        for i in range(n):
            if clusters[i] == 0:
                dists[0, i] = dists[0, i] + np.linalg.norm(X[i,:] - X[j,:], ord = 1)
            elif clusters[i] == 1:
                dists[1, i] = dists[1, i] + np.linalg.norm(X[i,:] - X[j,:], ord = 1)
            