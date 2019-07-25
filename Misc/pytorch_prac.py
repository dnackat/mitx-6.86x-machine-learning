#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:53:48 2019

@author: dileepn

PyTorch practice
"""
# Preliminaries
import torch
import numpy as np

#%% Tensors and their manipulation
a = torch.randn(5,5)
b = torch.randn(5,5)

matrix_mul = a.mm(b)    # Matrix multiplication
elem_wise = a * b   # Hadamard multiplication
mat_vec_mult = a.matmul(b[:,0])

numpy_ndarray = a.numpy()   # Convert to numpy array
back_to_torch = torch.from_numpy(numpy_ndarray)     # Conver numpy array back to tensor

another_tensor = a[2,2]     # Indexing
another_val = another_tensor.item()     # Convert to scalar

# Slicing examples
first_row = a[0,:]  
first_col = a[:,0]
combo = a[2:4, 2:4]

# In-place operations
a.add_(1)
a.div_(3)
a.zero_()

a = torch.randn(10,10)

# Manipulate dimensions
print(a.unsqueeze(-1).size())     # Add extra dim at the end
print(a.unsqueeze(1).size())    # Add extra dim at the beginning
print(a.unsqueeze(0).size())    # Add extra dim at the start
print(a.unsqueeze(-1).squeeze(-1).size())   # Undo extra dim
print(a.view(100,1).size())     # View things differently: flatten
print(a.view(50,2).size())     # View things differently: not flat

# Copy data to new dummy dimension
c = torch.randn(2)
print(c)
c = c.unsqueeze(-1)
print(c)
print(c.expand(2,3))

#%% Batching just adds an extra dimension (of size = batch_dim)
#a = torch.randn(10,5,5)
#b = torch.randn(10,5,5)
#
#c = a.bmm(b)    # Batch multiply
#print(c.size()) 

#%% Autograd: autmatic differentiation

# A tensor that will remember gradients
x = torch.randn(1, requires_grad = True)
print(x)
print(x.grad)

y = x.exp()
y.backward()    # Compute gradient
print(x.grad, y)

z = x*2
z.backward()
print(x.grad, z)    # Should be 2 but is 2 + e^x. Remember to zero out gradients

# Chain rule
x_a = torch.randn(1, requires_grad = True)
x_b = torch.randn(1, requires_grad = True)

x = x_a * x_b
x1 = x ** 2
x2 = 1/x1
x3 = x2.exp()
x4 = 1 + x3
x5 = x4.log()
x6 = x5 ** (1/3)

x6.backward()
print(x_a.grad)
print(x_b.grad)

x = torch.randn(1, requires_grad = True)
y = torch.tanh(x)
y.backward()
print(x.grad)