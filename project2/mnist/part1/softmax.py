import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

#pragma: coderesponse template
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    
    # Compute the matrix of theta*X' (each row is a category, column an example)
    R = (theta.dot(X.T))/temp_parameter
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    
    # Compute H matrix
    H = np.exp(R - c)
    
    # Divide H by the normalizing term
    H = H/np.sum(H, axis = 0)
    
    return H    
#pragma: coderesponse end

#pragma: coderesponse template
def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # avg error term
    
    # Clip prob matrix to avoid NaN instances
    clip_prob_matrix = np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15)
    
    # Take the log of the matrix of probabilities
    log_clip_matrix = np.log(clip_prob_matrix)
    
    # Create a sparse matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()
    
    # Only add terms of log(matrix of prob) where M == 1
    error_term = (-1/n)*np.sum(log_clip_matrix[M == 1])    
                
    # Regularization term
    reg_term = (lambda_factor/2)*np.linalg.norm(theta)**2
    
    return error_term + reg_term
    
#pragma: coderesponse end

#pragma: coderesponse template
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # Create spare matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    
    # Matrix of Probabilities
    P = compute_probabilities(X, theta, temp_parameter)
    
    # Gradient matrix of theta
    grad_theta = (-1/(temp_parameter*n))*((M - P) @ X) + lambda_factor*theta
    
    # Gradient descent update of theta matrix
    theta = theta - alpha*grad_theta
    
    return theta
    
#pragma: coderesponse end

#pragma: coderesponse template
def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    
    return (train_y_mod3, test_y_mod3)
#pragma: coderesponse end

#pragma: coderesponse template
def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    y_pred = get_classification(X, theta, temp_parameter)
    
    return 1 - (np.mod(y_pred, 3) == Y).mean()
#pragma: coderesponse end

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)

#%% Functions for kernelized softmax regression
def compute_kernel_probabilities(alpha_matrix, kernel_matrix, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        alpha_matrix - (k, n) NumPy array where row j represents alpha values for
                label j
        kernel_matrix - (n, n) NumPy array (similarity matrix, each column: phi(x_1).phi(x_1)
        to phi(x_n).phi(x_1))
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    
    # Compute the matrix of alpha*K*y
    R = alpha_matrix.dot(kernel_matrix)
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    
    # Compute H matrix
    H = np.exp(R - c)
    
    # Divide H by the normalizing term
    H = H/np.sum(H, axis = 0)
    
    return H

def compute_kernel_cost_function(alpha_matrix, kernel_matrix, Y, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        alpha_matrix - (k, n) NumPy array where row j represents alpha values for
                label j
        kernel_matrix - (n, n) NumPy array (similarity matrix, each column: phi(x_1).phi(x_1)
        to phi(x_n).phi(x_1))
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    
    # Get number of categories
    k = alpha_matrix.shape[0]
    
    # Get number of examples
    n = kernel_matrix.shape[0]
    
    ### avg error term ###
    
    # Clip prob matrix to avoid NaN instances
    clip_prob_matrix = np.clip(compute_kernel_probabilities(alpha_matrix, \
                            kernel_matrix, temp_parameter), 1e-15, 1-1e-15)
    
    # Take the log of the matrix of probabilities
    log_clip_matrix = np.log(clip_prob_matrix)
    
    # Create a sparse matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()
    
    # Only add terms of log(matrix of prob) where M == 1
    error_term = (-1/n)*np.sum(log_clip_matrix[M == 1])    
                
    ### Regularization term ###
    reg_term = (lambda_factor/2)*np.linalg.norm(alpha_matrix)**2
    
    return error_term + reg_term

def run_kernel_gradient_descent_iteration(alpha_matrix, kernel_matrix, Y, \
                            learning_rate, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        alpha_matrix - (k, n) NumPy array where row j represents alpha values for
                label j
        kernel_matrix - (n, n) NumPy array (similarity matrix, each column: phi(x_1).phi(x_1)
        to phi(x_n).phi(x_1))
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        learning_rate - the learning rate, alpha or eta (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        alpha - (k, n) NumPy array that is the final value of alpha
    """
    
    # Get number of labels
    k = alpha_matrix.shape[0]
    
    # Get number of examples
    n = kernel_matrix.shape[0]
    
    # Create spare matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    
    # Matrix of Probabilities
    P = compute_kernel_probabilities(alpha_matrix, kernel_matrix, temp_parameter)
    
    # Gradient matrix of theta
    grad_alpha = (-1/(temp_parameter*n))*((M - P) @ kernel_matrix) + lambda_factor*alpha_matrix
    
    # Gradient descent update of theta matrix
    alpha_matrix = alpha_matrix - learning_rate*grad_alpha
    
    return alpha_matrix

def softmax_kernel_regression(Y, kernel_matrix, temp_parameter, learning_rate, \
                              lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with alphas initialized to the all-zeros array. Here, alpha is a k by n NumPy array
    where row j represents the alpha values of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        kernel_matrix - (n, n) NumPy array (similarity matrix, each column: phi(x_1).phi(x_1)
        to phi(x_n).phi(x_1))
        temp_parameter - the temperature parameter of softmax function (scalar)
        learning_rate - the learning rate, alpha or eta (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        alpha - (k, n) NumPy array that is the final value of alpha
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    
    alphas = np.zeros([k, len(Y)])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_kernel_cost_function(alphas, kernel_matrix, \
                                    Y, lambda_factor, temp_parameter))
        alphas = run_kernel_gradient_descent_iteration(alphas, kernel_matrix, Y, learning_rate, \
                                                lambda_factor, temp_parameter)
    return alphas, cost_function_progression

def get_kernel_classification(alpha_matrix, kernel_matrix, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        kernel_matrix - (n, n) NumPy array (similarity matrix, each column: phi(x_1).phi(x_1)
        to phi(x_n).phi(x_1)). For the test set, kernel_matrix is (n,m) where m
        is the number of examples in the test set.
        alpha_matrix - (k, n) NumPy array where row j represents alpha values for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Predicted Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    
    probabilities = compute_kernel_probabilities(alpha_matrix, kernel_matrix, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def compute_kernel_test_error(alpha_matrix, kernel_matrix, Y, temp_parameter):
    assigned_labels = get_kernel_classification(alpha_matrix, kernel_matrix, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)