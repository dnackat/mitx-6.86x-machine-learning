import numpy as np

### Functions for you to fill in ###

# pragma: coderesponse template


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    kernel_matrix = (X.dot(Y.T) + c)**p
    
    return kernel_matrix
# pragma: coderesponse end

# pragma: coderesponse template


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n, d = X.shape
    m = Y.shape[0]

    # Naive version with for loops   
#    kernel_matrix = np.zeros((n,m))
#    
##### Single for loop ####
#    for i in range(n):
#        d = X[i,:] - Y  # Using numpy broadcasting to get differences
#        b = np.sum(d**2, axis=1)
#        kernel_matrix[i,:] = np.exp(-gamma*b)
#### End: single-loop version ####

#### Vectorized version (runs slower than the single loop version) ####
#    kernel_matrix = np.linalg.norm((X[:,None] - Y), ord=2, axis=2)**2
#    kernel_matrix = (np.sum(X**2, axis=1)[:,None] + np.sum(Y**2, axis=1)) - 2*(X @ Y.T)
    kernel_matrix = X**2 @ np.ones((d,m)) + np.ones((n,d)) @ Y.T**2 - 2*(X @ Y.T)
    kernel_matrix = np.exp(-gamma*kernel_matrix)
#### End: vectorized version ####
    
    return kernel_matrix
# pragma: coderesponse end
