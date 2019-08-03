#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:18:22 2019

@author: dileepn

Kernel Perceptron Algorithm: Toy Example
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data points
x = np.array([[5,2], [1,4], [2,0], [4,1], [3,3], [4,4], [1,1], [0,2], [5,5], [0,0]])
#x = np.array([[0,0], [0,2], [1,1], [2,0], [3,3], [1,4], [4,1], [4,4], [5,2], [5,5]])

# Labels
y = np.array([[1],[1],[-1],[1],[-1],[1],[-1],[-1],[1],[-1]])
#y = np.array([[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1]])

# Number of examples
n = x.shape[0]

# Number of features
m = x.shape[1]

# No. of iterations
T = 100

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
def shuffle(x, y, req='no'):
    """ Shuffles indices of data points to alter the order in which kernel
        perceptron algorithm runs through the data. 
        Inputs:
            Data (x and y)
            req: if 'yes', indices of x and y are randomly permuted; 
                 if 'no', x and y are returned as is.
        Returns: x and y
    """
    if req == 'yes':
       indices = np.random.permutation(len(x))
       x = x[indices,:]
       y = y[indices]
       return x, y
    else:
        return x, y

x, y = shuffle(x, y, 'no')
K = kernel_matrix(x,x)

# Start kernel perceptron loop
for t in range(T):
    counter = 0     # To check if all examples are classified correctly in loop
    for i in range(n):
        agreement = y[i]*(np.sum(alphas*y*(K[:,i].reshape(-1,1))) + theta0)
        if abs(agreement) < eps or agreement < 0.0:
            alphas[i] = alphas[i] + 1
            theta0 = theta0 + y[i]
        else:
            counter += 1
        
    # If all examples classified correctly, stop
    if counter == n:
        print("--------------------------------------------------------------")
        print("No. of iteration loops through the dataset:", t+1)
        print("--------------------------------------------------------------")
        break

# Initialize theta vector
theta = np.zeros((3,1))

# Calculate theta from calculated alphas
for i in range(n):
    theta = theta + alphas[i]*y[i]*quad_kernel(x[i,:])

print("theta0 =", theta0.item())
print("theta = [{:.2f}, {:.2f}, {:.2f}]".format(theta[0,0], theta[1,0], theta[2,0]))
print("----------------------------------------------------------------------")

print("=== Mistake Counts ===")
for i in range(n):
    print(x[i,:],":\t",alphas[i].item())

print("----------------------------------------------------------------------")
print("=== Classification Status ===")

for i in range(n):
    agreement = (y[i]*(theta.T.dot(quad_kernel(x[i,:])) + theta0)).item()
    if abs(agreement) < eps or agreement < 0.0:
        print("FAIL:\t", x[i,:])
    else:
        print("PASS:\t", x[i,:])

def decision_boundary(x, y, theta, theta0, space='orig'):
    """
    Computed the decision contour in original x-space. 
    
    Inputs: 
        x, y generated from np.meshgrid
        space in which to operate: original x-space (orig) or feature space (feature)
    Returns: Decision contour, theta.T.Phi(x) + theta0 
    """
    if space == 'feature':
        return (-theta0 - theta[0]*x - theta[1]*y)/theta[2]
    else:
        return theta[0]*x**2 + theta[1]*np.sqrt(2)*x*y + theta[2]*y**2  + theta0

# Plot in feature space
def plot_feature_space(theta, theta0):
    """
    Plots decision boundary in the transformed feature space.
    Inputs: 
        Trained parameter vector theta and offset theta0
    Returns: A 3D plot of the decision boundary along with the data points.
    """
    colors = ['r' if y == 1 else 'b' for y in y]
    
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    pts = quad_kernel(x).T  # Coordinates in feature space
    ax.scatter3D(pts[:,0], pts[:,1], pts[:,2], s=30, c=colors)
    
    xx, yy = np.meshgrid(np.linspace(*ax.get_xlim()), np.linspace(*ax.get_ylim()))
    zz = decision_boundary(xx, yy, theta, theta0, 'feature')   # Linear decision boundary in feature space
    
    ax.plot_surface(xx, yy, zz, cmap='winter', alpha=0.2)
    ax.view_init(elev=10, azim=60)
    ax.set_xlabel(r'$\Phi_1 = x_1^2$', fontsize=12)
    ax.set_ylabel(r'$\Phi_2 = \sqrt{2}x_1x_2$', fontsize=12)
    #ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$\Phi_3 = x_2^2$', fontsize=12)
    ax.set_title("Separating hyperplane in feature ($\Phi$-) space", fontsize=20)

# Plot decision boundary in original space
def plot_decision_boundary(theta, theta0, style='line'):
    """
    Plots decision boundary in the original x-space.
    Inputs: 
        Trained parameter vector theta and offset theta0
        Contour style: line or filled (levels = -1, 0, 1)
    Returns: A plot of the decision boundary along with the data points.
    """
    
    colors = ['r' if y == 1 else 'b' for y in y]
    
    f, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x[:,0], x[:,1], s=40, c=colors)
    xx, yy = np.meshgrid(np.linspace(*ax.get_xlim()), np.linspace(*ax.get_ylim()))
    z = decision_boundary(xx, yy, theta, theta0)
    
    if style == 'filled':
        cs = ax.contourf(xx, yy, z, levels=[-10,-5,0,5,10], 
                         colors = ['purple','cyan','orange','magenta'], 
                         extend='both', alpha=0.2)
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
        cs.clabel(cs.levels, inline=1, fontsize=10, colors='black', rightside_up=True)
        cs.changed()
        ax.set_title("Decision bounday in x-space (contour level = 0)", fontsize=20)
    else:
        cs = ax.contour(xx, yy, z, levels=[-10,-5,0,5,10], cmap='winter', 
                        alpha=0.5, linewidths=[1,1,2,1,1], 
                        linestyles=['dashed','dashed','solid','dashed','dashed'])
        cs.clabel(cs.levels, inline=1, fontsize=10)
        ax.set_title("Decision bounday in x-space (solid line)", fontsize=20)
    
    ax.set_xlabel(r'$x_1$', fontsize=20)
    ax.set_ylabel(r'$x_2$', fontsize=20)
    
plot_decision_boundary(theta, theta0, 'line')
plot_feature_space(theta, theta0)