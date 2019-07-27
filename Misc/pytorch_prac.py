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

#%% Manual Neural Net + Autograd SGD Example
import random
import matplotlib
import matplotlib.pyplot as plt

# Set our random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
# Get a simple dataset
from sklearn.datasets import make_classification

set_seed(7)

X, Y = make_classification(n_features = 2, n_redundant = 0, n_informative = 1,\
                           n_clusters_per_class = 1)

print("Number of examples: %d" % X.shape[0])
print("Number of features: %d" % X.shape[1])

# Take a peek at the data
plt.scatter(X[:,0], X[:,1], marker='o', c=Y, s=25, edgecolor='k')
plt.show()

# Convert data to PyTorch
X, Y = torch.from_numpy(X), torch.from_numpy(Y)
X, Y = X.float(), Y.float()

# 1 layer neural net to classify this dataset

# Define dims
num_features = 2
hidden_size = 100
num_outputs = 1

# Learning rate
eta = 0.1
num_steps = 1000

# Input to hidden weights
W1 = torch.randn(hidden_size, num_features, requires_grad = True)
b1 = torch.zeros(hidden_size, 1, requires_grad = True)

# Hidden to output weights
W2 = torch.randn(num_outputs, hidden_size, requires_grad = True)
b2 = torch.zeros(num_outputs, 1, requires_grad = True)

# Group params
params = [W1, b1, W2, b2]

# Get random order
indices = torch.randperm(X.size(0))

# Keep running average losses for a learning curve
avg_loss = []

# Run 
for step in range(num_steps):
    # Get example
    i = indices[step % indices.size(0)]
    x_i, y_i = X[i], Y[i]
    x_i = x_i.view(x_i.size(0),1)

    # Run example
    hidden = torch.relu(W1.matmul(x_i) + b1)
    y_hat = torch.sigmoid(W2.matmul(hidden) + b2)
    
    # Compute loss binary cross entropy: -(y_i * log(y_hat) + (1 - y_i) * log(1 - y_hat))
    # Epsilon for numerical stability
    eps = 1e-6
    loss = -(y_i * (y_hat + eps).log() + (1 - y_i) * (1 - y_hat + eps).log())
    
    # Add to our running average learning curve. Don't forget .item()!
    if step == 0:
        avg_loss.append(loss.item())
    else:
        old_avg = avg_loss[-1]
        new_avg = (loss.item() + old_avg * len(avg_loss))/(len(avg_loss) + 1)
        avg_loss.append(new_avg)
        
    # Zero out all previous gradients
    for param in params:
        # It might start out as zero
        if param.grad is not None:
            param.grad.zero_()
            
    # Backward pass
    loss.backward()
    
    # Update parameters
    for param in params:
        # In-place
        param.data = param.data - eta * param.grad
        
# Plot
plt.plot(range(num_steps), avg_loss)
plt.ylabel('Loss')
plt.xlabel('Step')
plt.show()

#%% torch.nn package 
import torch.nn as nn
import torch.nn.functional as F

# Linear layer: in_features, out_features
linear = nn.Linear(10, 10)
print(linear)

# Convolution layer: in_channels, out_channels, kernel_size, stride
conv = nn.Conv2d(1, 20, 5, 1)
print(conv)

# RNN: num_inputs, num_hidden, num_layers
rnn = nn.RNN(10, 10, 1)
print(rnn)

print(linear.weight)
print([k for k, v in conv.named_parameters()])

#%% Make our own model!
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        
        # 1 input channel to 20 feature maps of 5x5 kernel with stride 1.
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        
        # 20 input channels to 50 feature maps of 5x5 kernel with stride 1.
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
        # Fully connected of final 4x4 image to 500 features
        self.fc1 = nn.Linear(4*4*50, 500)
        
        # From 500 to 10 classes
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
# Initialize the model
model = NNet()

# Optimizer
import torch.optim as optim
# Initialize the modelalize with model params
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Full train and test loops
import tqdm

def train(model, train_loader, optimizer, epoch):
    # For things like dropout
    model.train()
    
    # Avg loss
    total_loss = 0
    
    # Iterate through dataset
    for data, target in tqdm.tqdm(train_loader):
        # Zero grad
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Negative log likelihood loss func
        loss = F.nll_loss(output, target)
        
        # Backward pass
        loss.backward()
        total_loss += loss.item()
        
        # Update
        optimizer.step()
        
    # Print average loss
    print("Train Epoch: {}\t Loss: {:.6f}".format(epoch, total_loss/len(train_loader)))
    
def test(model, test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()     # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)   # Get the index of the max log-probabiity
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_loss /= len(test_loader.dataset)
            
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
                  .format(test_loss, correct, len(test_loader.dataset), \
                          100. * correct/len(test_loader.dataset)))
            
# Time to run MNIST
from torchvision import datasets, transforms

# See the torch Dataloader for more details
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=32, shuffle=True)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=True)