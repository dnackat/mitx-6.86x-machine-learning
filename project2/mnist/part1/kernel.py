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
    n = X.shape[0]
    m = Y.shape[0]

# Naive version with for loops   
#    kernel_matrix = np.zeros((n,m))
#    
#    for i in range(n):
#        for j in range(m):
#            kernel_matrix[i,j] = np.exp(-gamma*(np.linalg.norm(X[i,:] - Y[j,:])**2))

# Vectorized version
    XdiffY = np.repeat(X,m,axis=0) - np.repeat(Y,n,axis=0)
    
    kernel_matrix = np.exp(-gamma*np.sum(XdiffY**2,axis=1))
            
    return kernel_matrix
# pragma: coderesponse end
