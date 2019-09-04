#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:44:03 2019

@author: dileepn

Using ScikitLearn for multiple linear regression in order to predict house prices  
"""
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.regression import mean_squared_error 
import matplotlib.pyplot as plt

# Open the dataset and define X, y, and m
def load_data():
    """ This function loads: 
        1. Space-separated text (*.txt) file using numpy
        and returns X and y as numpy arrays. Sample input (last column is y):
            2 100
            0.44 0.68 511.14
            0.99 0.23 717.1
            0.84 0.29 607.91
            0.28 0.45 270.4
            0.07 0.83 289.88
                    .
                    .
                    .
            4
            0.05 0.54
            0.91 0.91
            0.31 0.76
            0.51 0.31
            
                    OR
        
        2. Data entered manually (space-separated) on the standard input 
        and stores them in X and y. """

    while True:
        
        # Prompt user for dataset input type
        input_type = input("Choose dataset input type. 1 (file) or 2 (manual entry): ")
    
        if input_type == "1":
            
            # Prompt for filepath
            filepath = input("Enter the complete filepath (/home/user...): ")
            
            # Temporary lists to store data as it is being read
            temp_data = []
            temp_test_data = []

            # Read the dataset line-by-line. Get num. of features, F, and 
            # num. of examples, N
            with open(filepath) as input_file:
            
                for line_num, line in enumerate(input_file):
                    if line_num == 0:
                        F, N = line.split()
                        F, N = int(F), int(N)
                    elif line_num == N + 1:
                        T = int(line)
                    elif line_num > 0 and line_num <= N:
                        x1, x2, y = line.split()
                        # Store as ordered pair in temp_data
                        temp_data += [(float(x1), float(x2), float(y))]
                    elif line_num > N + 1 and line_num <= N + T + 1:
                        x1, x2 = line.split()
                        temp_test_data += [(float(x1), float(x2))]
                    
            # Convert temp lists into numpy arrays
            dataset = np.array(temp_data)
            X_pred = np.array(temp_test_data)       
            
            # Define X, y, and m
            X = dataset[:, :F]
            y = dataset[:, F].reshape(-1,1)

            break
            
        elif input_type == "2":
            
            # First line has number of features and number of training examples
            F, N = map(int, input().split())
            
            # Get the training set (X and y)
            train = np.array([input().split() for _ in range(N)], dtype=np.float64)
            
            # Number of test examples
            T = int(input())
            X_pred = np.array([input().split() for _ in range(T)], dtype=np.float64)
            
            # Split the training set into X and y
            X = train[:,:F]
            y = train[:,F]
            
            break
        
        else:
            print("Incorrect input. Please enter 1 or 2.")
    
    return (X, y, X_pred)

# Load data
X_train, y_train, X_test = load_data()

# Fit the model
model = linear_model.LinearRegression()

#%% Linear regression
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Error metrics
y_test = np.array([105.22, 142.68, 132.94, 129.71])
mse = mean_squared_error(y_test, y_pred)

print("MSE = {:.2f}".format(mse))

#%% Now with polynomial features
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
model.fit(X_poly, y_train)
y_pred_poly = model.predict(X_test_poly)

for i in range(len(y_pred_poly)):
    print("{:.2f}".format(y_pred_poly[i].item()))
