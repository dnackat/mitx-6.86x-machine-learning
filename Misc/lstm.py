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
W_oh = 0, W_ox = 100, b_o = 0   # output gate
W_ch = -100; W_cx = 50; b_c = 0     # memory cell

# Length of sequence
n = 6   

# Vectors of states and gates
f_t = np.zeros((n+1,1))     # forget gate
i_t = np.zeros((n+1,1))     # input gate
o_t = np.zeros((n+1,1))     # output gate
c_t = np.zeros((n+1,1))     # vector of memory cells
h_t = np.zeros((n+1,1))     # vector of visible states

# Vector of input sequence 
x_t = np.array([[0],[0],[1],[1],[1],[0]])

# Sigmoid function
def sigmoid(x):
    """ Computes the sigmoid output (1/(1+e^-x)) of a number, x. and rounds to
        the nearest integer. """
        
    sig = 1/(1+np.exp(-x))
    return np.floor(sig + 0.5)

# Tan activation with rounding to nearest integer
def tan_act(x):    
    """ Computes"""
# Compute gate and output values sequentially
for i in range(n):
    