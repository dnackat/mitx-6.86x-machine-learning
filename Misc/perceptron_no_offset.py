#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:57:13 2019

@author: dileepn

Perceptron algorithm without offset (hyperplane through origin)
"""
import numpy as np
import matplotlib.pyplot as plt

# Data points
#x = np.array([[-1,-1],[1,0],[-1,10]])
x = np.array([[1,0],[-1,10],[-1,-1]])

# Labels
#y = np.array([[1],[-1],[1]])
y = np.array([[-1],[1],[1]])

# Number of examples
n = x.shape[0]

# Number of features
m = x.shape[1]

# No. of iterations
T = 10

# Initialize parameter vector
theta = np.array([[0],[0]])

# Start the perceptron update loop
mistakes = 0    # Keep track of mistakes
for t in range(T):
    counter = 0     # To check if all examples are classified correctly in loop
    for i in range(n):
        if float(y[i]*(theta.T.dot(x[i,:]))) <= 0:
            theta += y[i]*x[i,:].reshape((m,1))
            print("current parameter vector:", theta)
            mistakes += 1
        else:
            counter += 1
    
    # If all examples classified correctly, stop
    if counter == n:
        break
    
# Print total number of mistakes
print("Total number of mistakes:", mistakes)