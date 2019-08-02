#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:18:22 2019

@author: dileepn

Kernel Perceptron Algorithm: Toy Example
"""
import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([[0,0],[0,2],[1,1],[2,0],[3,3],[1,4],[4,1],[4,4],[5,2],[5,5]])

# Labels
y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])

# Plot data
colors = ['r' if y == 1 else 'b' for y in y]
f, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x[:,0], x[:,1], s=40, c=colors)

# Number of examples
n = x.shape[0]

# Number of features
m = x.shape[1]

# No. of iterations
T = 10

# Initialize alphas to zero
alphas = np.zeros((n,1))
#alphas = np.array([[1],[31],[11],[65],[72],[21],[30],[4],[0],[15]])
theta0 = 0.0

# Tolerance for floating point errors
eps = 1e-8

# Kernel matrix: quadratic kernel, phi(x) = [x1^2, sqrt(2)*x1*x2, x2^2]
def kernel_matrix(x, y):
    """ 
    Returns kernel matrix (quadratic kernel, in this case) for input arrays, x 
    and y: K(x,y) = phi(x).phi(y), which for a quadratic kernel is (x.y)^2 
    """
    K = (x.dot(y.T))**2
    
    return K

def quad_kernel(x):
    """
    Returns the kernel representation (quadratic, in this case) of input array, x
    """
    try:
        x.shape[1]
        k1 = x[:,0]**2
        k2 = x[:,1]**2
        k12 = np.sqrt(2)*x[:,0]*x[:,1]
        return np.vstack((k1,k12,k2))
    except IndexError:
        k1 = x[0]**2
        k2 = x[1]**2
        k12 = np.sqrt(2)*x[0]*x[1]
        return np.array([[k1],[k12],[k2]])

# Kernel matrix for our case
indices = np.random.permutation(len(x))
np.random.shuffle(indices)
x = x[indices,:]
y = y[indices]
K = kernel_matrix(x,x)

# Start kernel perceptron loop
for t in range(T):
    counter = 0     # To check if all examples are classified correctly in loop
    for i in range(n):
        agreement = float(y[i]*np.sum(alphas.dot(y.T).dot(K[:,i]), axis=0))
        if abs(agreement) < eps or agreement < 0.0:
            alphas[i] = alphas[i] + 1
            theta0 = theta0 + y[i]
        else:
            counter += 1
        
    # If all examples classified correctly, stop
    if counter == n:
        print("No. of iteration loops through the dataset:", t+1)
        break

theta = np.zeros((3,1))

for i in range(n):
    theta = theta + alphas[i]*y[i]*quad_kernel(x[i,:])
    #theta0 = theta0 + float(alphas[i]*y[i])

print("theta0=", theta0)
print("theta=", theta)

for i in range(n):
    if theta.T.dot(quad_kernel(x[i,:])) + theta0 > 0.0:
        print("PASS")
    else:
        print("FAIL")

def decision_boundary(x, theta, theta0):
    return theta.T.dot(quad_kernel(x)) + theta0

