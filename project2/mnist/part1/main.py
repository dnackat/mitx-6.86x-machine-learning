import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, or the below functions will not work

def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
#print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the below functions will not work

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: first fill out functions in softmax.py, or run_softmax_on_MNIST will not work

def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    
    # Update labels to label mod 3
    #train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Compute test error with updated labels
    #test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    
    return test_error

#print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1.0))

# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

#######################################################################
# 6. Changing Labels
#######################################################################

#pragma: coderesponse template
def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y_mod3, theta, temp_parameter)
    return test_error

#print('softmax test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1.0))
#pragma: coderesponse end

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.

#n_components = 18
#pcs = principal_components(train_x)
#train_pca = project_onto_PC(train_x, pcs, n_components)
#test_pca = project_onto_PC(test_x, pcs, n_components)
# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.


# TODO: Train your softmax regression model using (train_pca, train_y)
#       and evaluate its accuracy on (test_pca, test_y).
#theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter=1, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)

#plot_cost_function_over_time(cost_function_history)

# Compute the test error with projected test data
#test_error = compute_test_error(test_pca, test_y, theta, temp_parameter=1)

#print("Test error with 18-dim PCA representation:", test_error)

# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
#plot_PC(train_x[range(100),], pcs, train_y[range(100)])


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
#firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x)
#plot_images(firstimage_reconstructed)
#plot_images(train_x[0,])
#
#secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x)
#plot_images(secondimage_reconstructed)
#plot_images(train_x[1,])



## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set
#n_components = 10
#pcs = principal_components(train_x)
#train_pca10 = project_onto_PC(train_x, pcs, n_components)
#test_pca10 = project_onto_PC(test_x, pcs, n_components)
#
## TODO: First fill out cubicFeatures() function in features.py as the below code requires it.
#
#train_cube = cubic_features(train_pca10)
#test_cube = cubic_features(test_pca10)
## train_cube (and test_cube) is a representation of our training (and test) data
## after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.
#
#
## TODO: Train your softmax regression model using (train_cube, train_y)
##       and evaluate its accuracy on (test_cube, test_y).
#theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter=1, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
#
#plot_cost_function_over_time(cost_function_history)
#
#test_error = compute_test_error(test_cube, test_y, theta, temp_parameter=1)
#
#print("Test error with 10-dim PCA with cubic features:", test_error)

#%% Implement kernelized softmax regression
    
# Truncate the training set to have only 20,000 examples
import time

# Start time
start_time = time.time()

n = 100
n_test = 20
k = 10  # number of categories
indices_train = np.random.permutation(n)
indices_test = np.random.permutation(n_test)

train_x_trunc = train_x[indices_train,:]
train_y_trunc = train_y[indices_train]
test_x_trunc = test_x[indices_test,:]
test_y_trunc = test_y[indices_test]

# Find PCA representation of training and test sets
n_components = 18
pcs = principal_components(train_x_trunc)
train_pca = project_onto_PC(train_x_trunc, pcs, n_components)
test_pca = project_onto_PC(test_x_trunc, pcs, n_components)

# Kernel matrix for training data
kernel_train = rbf_kernel(train_pca, train_pca, gamma=0.5)

# Kernel matrix for test data
kernel_test = rbf_kernel(train_pca, test_pca, gamma=0.5)
    
def run_kernel_softmax_on_MNIST(kernel_train, train_y, kernel_test, test_y, \
                                temp_parameter=1.0, lambda_factor=0.01, \
                                k=10, learning_rate=0.3, num_iterations=150):

    # Softmax kernel regression
    alphas, cost_function_history = softmax_kernel_regression(train_y, kernel_train, \
                                    temp_parameter, learning_rate, \
                                    lambda_factor, k, num_iterations)
    # Plot cost vs iteration
    plot_cost_function_over_time(cost_function_history)
    
    # Calculate test error
    test_error_kernel = compute_kernel_test_error(alphas, kernel_test, test_y, temp_parameter=1)
    
    return test_error_kernel

print("Test error for kernelized softmax regression:", \
      run_kernel_softmax_on_MNIST(kernel_train, train_y_trunc, kernel_test, test_y_trunc, \
                                temp_parameter=0.5, lambda_factor=0.01, \
                                k=10, learning_rate=0.3, num_iterations=150))

# End time
end_time = time.time()

print("-----------------------------------------------------------------")
print("Time taken to run kernelized softmax regression with {} training examples\
 and {} test examples with {} categories and {} principal components\
 is: {:.2f} s".format(n, n_test, k, n_components,end_time-start_time))
print("-----------------------------------------------------------------")
   
#%% Kernelized classification using sklearn's SVM package
#import sklearn
#svm = sklearn.svm.SVC(C=100, gamma='scale')
#svm.fit(train_x, train_y)
#print((test_y == svm.predict(test_x)).mean())

#%% Check gradient
h = 1e-4
alpha_matrix = np.zeros([k,n])

grad = compute_kernel_gradient(alpha_matrix, kernel_train, train_y_trunc, \
                               lambda_factor=1e-5, temp_parameter=0.5)

num_grad = np.zeros(grad.shape)

for i in range(k):
    for j in range(n):
        alpha_more = np.zeros(alpha_matrix.shape)
        alpha_more[i,j] = alpha_more[i,j] + h
        cost_more = compute_kernel_cost_function(alpha_more, kernel_train, train_y_trunc, lambda_factor=0.01, temp_parameter=0.5)
        alpha_less = np.zeros(alpha_matrix.shape)
        alpha_less[i,j] = alpha_less[i,j] - h
        cost_less = compute_kernel_cost_function(alpha_less, kernel_train, train_y_trunc, lambda_factor=0.01, temp_parameter=0.5)
        num_grad[i,j] = (cost_more-cost_less)/(2*h)

relative_error = np.max(np.abs((grad-num_grad)/np.maximum(np.abs(grad), np.abs(num_grad))))

print("The relative error for gradient computation is:", relative_error)