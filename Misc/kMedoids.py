#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:22:17 2019

@author: dileepn

k-Medoids: A Toy Example 
"""
import numpy as np

# Data points as a matrix
X = np.array([[0,-6],[4,4],[0,0],[5,-2]])

# Initialize representatives
z = np.array([[-5,2],[0,-6]])

# Cluster affiliations
clusters = np.ones((X.shape[0], 1))


