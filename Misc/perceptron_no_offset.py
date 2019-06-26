#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:57:13 2019

@author: dileepn

Perceptron algorithm without offset (hyperplane through origin)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data points
x = np.array([[np.cos(np.pi),0,0],[0,np.cos(2*np.pi),0],[0,0,np.cos(3*np.pi)]])
#x = np.array([[0,np.cos(2*np.pi)],[np.cos(np.pi),0]])
#x = np.array([[1,0],[-1,10],[-1,-1]])

# Labels
y = np.array([[1],[1],[1]])
#y = np.array([[-1],[1],[1]])

# Plot data
colors = ['b' if y == 1 else 'r' for y in y]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:,1], x[:,2], s=40, c=colors)
ax.hold(True)
#plt.plot(x[1,0], x[1,1], color='blue', marker='o', markersize=8, ls = '')

# Number of examples
n = x.shape[0]

m = x.shape[1]

# No. of iterations
T = 10

# Initialize parameter vector
theta = np.zeros((m,1))

# Tolerance for floating point errors
eps = 1e-5

# Start the perceptron update loop
mistakes = 0    # Keep track of mistakes
for t in range(T):
    counter = 0     # To check if all examples are classified correctly in loop
    for i in range(n):
        agreement = float(y[i]*(theta.T.dot(x[i,:])))
        if abs(agreement) < eps or agreement < 0.0:
            theta = theta + y[i]*x[i,:].reshape((m,1))
            print("current parameter vector:", theta)
            mistakes += 1
        else:
            counter += 1
    
    # If all examples classified correctly, stop
    if counter == n:
        print("No. of iteration loops through the dataset:", t+1)
        break
    
# Print total number of mistakes
print("Total number of misclassifications:", mistakes)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
z_line = (- theta[0]*xx - theta[1]*yy)/theta[2]
ax.plot_surface(xx, yy, z_line, alpha = 0.2)