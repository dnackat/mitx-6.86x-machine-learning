#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:57:56 2019

@author: dileepn

Feed-forward neural network with one hidden layer: Toy example
"""
import numpy as np
import matplotlib.pyplot as plt

# Input vector
x = np.array([[3],[14],[1]])    # Last element is for the offset

# Weights matrix, W
W = np.array([[1,0,-1],[0,1,-1],[-1,0,-1],[0,-1,-1]])

# Weights matrix, V
V = np.array([[1,1,1,1,0],[-1,-1,-1,-1,2]])

def calc_output(input_vector, weights1, weights2):
    """ Implements a simple feed-forward neural network with one hidden layer
    and returns the output vector. ReLU activation is used in all layers. """
    # Transform the input vector
    z = weights1.dot(input_vector)
    
    # Use ReLU activation function
    z[z < 0] = 0    # This is f(z)
    
    # Append one to f(z) to account for the bias term
    z = np.append(z, 1)
    
    # Transform f(z) to get the signals at the output layer
    u = V.dot(z)
    
    # Apply ReLU activation function at the output layer to get f(u)
    u[u < 0] = 0
    
    # Get the final output by applying the softmax function to u
    o = softmax(u)
    
    return o
    
def softmax(z):
    """ Applies the softmax function: exp(z)/sum(exp(z)) to an input vector z"""
    
    scores = np.exp(z)
    norm_constant = np.sum(scores)
    
    return scores/norm_constant

# Compute output for our data
output = calc_output(x, W, V)

print(output)