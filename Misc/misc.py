#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:55:22 2019

@author: dileepn
"""
#%% Plot data and draw decision boundaries
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([[-1,1],[1,-1],[1,1],[2,2]])   # data points
y = np.array([[1],[1],[-1],[-1]])   # labels

t = np.linspace(-2,2,100)
r = 2.5     # radius
o1 = -1     # origin x
o2 = -1     # origin y
x1 = o1 + r*np.cos(t)   # Circle x
y1 = o2 + r*np.sin(t)   # Circle y
x_line = np.linspace(-2,2,100)
theta1 = -1    # offset
theta2 = -1   # parameter 1
theta0 = 1  # parameter 2
y_line = (-theta0 - theta1*x_line)/theta2

# Plot
plt.figure()
plt.plot(x[:2,0], x[:2,1], color='red', marker='+', markersize=10, ls = '')
plt.plot(x[2:,0], x[2:,1], color='blue', marker='o', markersize=8, ls = '')
#plt.plot(x1,y1,'g-',linewidth=2,label='decision boundary')
plt.plot(y_line,x_line,'g-',linewidth=2,label='decision boundary')
plt.legend()