#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:41:44 2019

@author: dileepn

LSTM: A simple toy example
"""
import numpy as np
import matplotlib.pyplot as plt

# Weights and bias
W_fh = 0; W_fx = 0; b_f = -100  # forget gate
W_ih = 0; W_ix = 100; b_i = 100     # input gate
W_oh = 0; W_ox = 100; b_o = 0   # output gate
W_ch = -100; W_cx = 50; b_c = 0     # memory cell

# Length of sequence
n = 6   

# Vectors of states and gates
f_t = np.zeros((n,1))     # forget gate
i_t = np.zeros((n,1))     # input gate
o_t = np.zeros((n,1))     # output gate
c_t = np.zeros((n,1))     # vector of memory cells
h_t = np.zeros((n,1))     # vector of visible states

# Vector of input sequence 
x_t = np.array([[0],[0],[1],[1],[1],[0]])

# Sigmoid function
def sigmoid(x):
    """ Computes the sigmoid output (1/(1+e^-x)) of a number, x. """
    
    if x >= 50:
        return 1.0
    elif x <= -50:
        return 0.0
    
    sig = 1/(1 + np.exp(-x))
    return sig

# Compute gate and output states sequentially
for i in range(n):
    if i == 0:
        h = 0.0
        c = 0.0
    else:
        h = float(h_t[i-1])
        c = float(c_t[i-1])
        
    # Current input, x_t
    x = float(x_t[i])
    
    # Gate values
    f_t[i] = sigmoid(W_fh*h + W_fx*x + b_f)     # forget gate
    i_t[i] = sigmoid(W_ih*h + W_ix*x + b_i)     # input gate
    o_t[i] = sigmoid(W_oh*h + W_ox*x + b_o)     # output gate

    # Revised memory cell values, C'
    c_rev = np.tanh(W_ch*h + W_cx*x + b_c)

    # New memory cell state (what to forget + what to remember)
    c_t[i] = f_t[i]*c + i_t[i]*c_rev
    
    # New visible state rounded to the nesrest integer (what to show)
    h_t[i] = np.floor(o_t[i]*np.tanh(c_t[i]) + 0.5)
    
print(h_t)