"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    mu, var, pi = mixture   # Unpack mixture tuple
    K = mu.shape[0]
    
    # f(u,j) matrix that's used to store the normal matrix and log of posterior probs: (p(j|u))
    f = np.zeros((n,K), dtype=np.float64)
    
    # Compute the normal matrix: Single loop implementation
    for i in range(n):
        # For each user pick only columns that have ratings
        Cu_indices = X[i,:] != 0
        # Dimension of Cu (no. of non-zero entries)
        dim = np.sum(Cu_indices)
        # log of pre-exponent for this user's gaussian dist.
        pre_exp = (-dim/2.0)*np.log((2*np.pi*var))
        # Calculate the exponent term of the gaussian
        diff = X[i, Cu_indices] - mu[:, Cu_indices]    # This will be (K,|Cu|)
        norm = np.sum(diff**2, axis=1)  # This will be (K,)
        
        # Now onto the final log normal matrix: log(N(...))
        # We will need log(normal), exp will cancel, so no need to calculate it
        f[i,:] = pre_exp - norm/(2*var)  # This is the ith users log gaussian dist vector: (K,)
    
    f = f + np.log(pi + 1e-16)  # This is the f(u,j) matrix
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)  # Store this to calculate log_lh
    log_posts = f -  logsums # This is the log of posterior prob. matrix: log(p(j|u))
    
    log_lh = np.sum(logsums, axis=0).item()   # This is the log likelihood
    
    return np.exp(log_posts), log_lh


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    
    # Calculate revised pi(j): same expression as in the naive case
    pi = np.sum(post, axis=0)/n
    
    # Create delta matrix indicating where X is non-zero
    delta = X.astype(bool).astype(int)
    
    # Update means only when sum_u(p(j|u)*delta(l,Cu)) >= 1
    


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
