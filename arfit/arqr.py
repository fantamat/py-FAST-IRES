"""
ARfit: QR factorization for least squares estimation of AR model.

This module implements the QR factorization needed for least squares estimation of AR models.
"""

import numpy as np
from scipy import linalg

def arqr(v, p, mcor):
    """
    QR factorization for least squares estimation of AR model.
    
    Parameters
    ----------
    v : array_like
        Time series data. If v is a matrix, columns represent variables.
        If v has three dimensions, the third dimension corresponds to trials.
    p : int
        Order of the AR model.
    mcor : int
        If 1, a vector of intercept terms is being fitted.
        If 0, the process v is assumed to have mean zero.
    
    Returns
    -------
    R : ndarray
        Upper triangular matrix appearing in the QR factorization of the AR model.
    scale : ndarray
        Vector of scaling factors used to regularize the QR factorization.
    
    Notes
    -----
    This function computes the QR factorization needed in the least squares estimation
    of parameters of an AR(p) model.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    """
    # Get dimensions
    if v.ndim == 2:
        n, m = v.shape
        ntr = 1
        v = v.reshape(n, m, 1)  # Add singleton dimension for trials
    else:
        n, m, ntr = v.shape  # n time steps, m variables, ntr trials
    
    ne = ntr * (n - p)  # Number of block equations of size m
    np_param = m * p + mcor  # Number of parameter vectors of size m
    
    # Initialize the data matrix K (of which a QR factorization will be computed)
    K = np.zeros((ne, np_param + m))
    if mcor == 1:
        # First column of K consists of ones for estimation of intercept vector w
        K[:, 0] = np.ones(ne)
    
    # Assemble 'predictors' u in K
    for itr in range(ntr):
        for j in range(1, p + 1):
            idx = slice((n - p) * itr, (n - p) * (itr + 1))
            col_idx = slice(mcor + m * (j - 1), mcor + m * j)
            K[idx, col_idx] = v[p-j:n-j, :, itr]
        
        # Add 'observations' v (left hand side of regression model) to K
        K[(n - p) * itr:(n - p) * (itr + 1), np_param:np_param + m] = v[p:n, :, itr]
      # Compute regularized QR factorization of K:
    # The regularization parameter delta is chosen according to Higham's (1996)
    # Theorem 10.7 on the stability of a Cholesky factorization
    q = np_param + m  # Number of columns of K
    delta = (q**2 + q + 1) * np.finfo(float).eps
    scale = np.sqrt(delta) * np.sqrt(np.sum(K**2, axis=0))
    
    # Compute the QR factorization using scipy.linalg.qr
    K_padded = np.vstack((K, np.diag(scale)))
    # Mode 'r' returns a tuple in scipy.linalg.qr, so extract the R matrix
    R = linalg.qr(K_padded, mode='r')
    if isinstance(R, tuple):
        R = R[0]  # Extract R matrix from the tuple
    
    return R, scale
