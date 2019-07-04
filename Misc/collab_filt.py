#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:44:03 2019

@author: dileepn

HW3: Predictions using Collaborative Filtering (rank 1 assumed)
"""
import numpy as np

# Incomplete rating matrix (-1 = unfilled)
Y = np.array([[5.0,-1.0,7.0],[-1.0,2.0,-1.0],[4.0,-1.0,-1.0],[-1.0,3.0,6.0]])

# Rating mtrix to be predicted
X = np.zeros(Y.shape)

# Initialize matrix factor, V
V = np.array([[4.0],[2.0],[1.0]])

# Matrix factor, U
U = np.array([[6.0],[0.0],[3.0],[6.0]])

# Compute rating matrix with this initialization
X = U @ V.T

# Compute Least squares error term in the objective function
D = Y - X
D[Y < 0] = 0
ls_error = np.sum((D**2))/2

print("Least squares error:", ls_error)

# Regularization parameter
lam = 1

# Compuate regularization term in the objective function
reg_term = lam*((np.linalg.norm(U))**2 + (np.linalg.norm(V))**2)/2
print("Regularization term is:", reg_term)

# Update loop for U with V fixed
for a in range(Y.shape[0]):
    Y_sum = 0.0
    v_sum = 0.0
    for i in range(Y.shape[1]):
        if Y[a,i] >= 0:
            Y_sum = Y_sum + float(Y[a,i]*V[i])
            v_sum = v_sum + float(V[i]**2)
    U[a] = Y_sum/(lam + v_sum)
    
print("Updated U is:", U)