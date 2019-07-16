#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 02:00:08 2019

@author: dileepn

Feedforward Neural Network: Toy Example
"""
import numpy as np
import matplotlib.pyplot as plt

num_pts = 4

def calc_activation(x,w,bias,fn="linear"):
    z = x.dot(w) + bias.T
    if fn == "linear":
        f = 5*z - 2
    elif fn == "ReLU":
        z[z <= 0] = 0
        f = z
    elif fn == "tanh":
        f = np.tanh(z)
    return f

x = np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
y = np.array([[1],[-1],[-1],[1]])

bias = np.array([[1],[1]])
w = np.array([[1,-1],[-1,1]]) 

f = calc_activation(x,w,bias,"ReLU")
print(f)

# Plot data
colors = ['b' if y == 1 else 'r' for y in y]
plt.figure()
plt.scatter(f[:,0], f[:,1], s=40, c=colors)