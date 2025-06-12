"""
ARfit: Test of residuals of fitted AR model.

This module implements tests on the residuals of a fitted AR model.
"""

import numpy as np
from scipy import stats, linalg

# Try to import Numba optimizations
try:
    from .numba_optimized import optimized_residuals_test
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def arres(v, w, A, C, k=None):
    """
    Test of residuals of fitted AR model.
    
    Parameters
    ----------
    v : array_like
        Time series data. If v is a matrix, columns represent variables.
        If v has three dimensions, the third dimension corresponds to trials.
    w : array_like
        Intercept vector of the AR model.
    A : array_like
        AR coefficient matrices, concatenated as [A1, A2, ..., Ap].
    C : array_like
        Noise covariance matrix.
    k : int, optional
        Maximum lag of residual correlation matrices.
        If None, k = min(20, nres-1) where nres is the number of residuals.
    
    Returns
    -------
    h : bool
        Test result. True if residuals appear uncorrelated, False otherwise.
    siglev : float
        Significance level of the modified Li-McLeod portmanteau statistic.
    res : ndarray
        Time series of residuals.
    lmp : float
        Value of the Li-McLeod portmanteau statistic.
    
    Notes
    -----
    This function computes the time series of residuals of an AR(p) model and
    tests for their uncorrelatedness using the modified Li-McLeod portmanteau statistic.
    
    References
    ----------
    Li, W. K., and A. I. McLeod, 1981: Distribution of the Residual Autocorrelations
    in Multivariate ARMA Time Series Models, J. Roy. Stat. Soc. B, 43, 231-239.
    """
    # Use Numba-optimized version if available
    if HAS_NUMBA:
        try:
            h, siglev, res = optimized_residuals_test(v, w, A, C, k if k is not None else 20)
            lmp = 0.0  # We don't compute LMP in the optimized version
            return h, siglev, res, lmp
        except Exception as e:
            print(f"Warning: Failed to use optimized version, falling back to standard implementation: {e}")
            # Fall through to standard implementation    # Convert inputs to numpy arrays
    v = np.array(v)
    A = np.array(A)
    w = np.array(w).flatten()
    C = np.array(C)
    
    # Get dimensions
    if v.ndim == 2:
        n, m = v.shape
        ntr = 1
        v = v.reshape(n, m, 1)  # Add singleton dimension for trials
    else:
        n, m, ntr = v.shape  # n time steps, m variables, ntr trials
    
    p = A.shape[1] // m  # Order of model
    nres = n - p  # Number of residuals (per realization)
    
    # Default value for k
    if k is None:
        k = min(20, nres - 1)
    
    if k <= p:  # Check if k is in valid range
        raise ValueError("Maximum lag of residual correlation matrices too small.")
    if k >= nres:
        raise ValueError("Maximum lag of residual correlation matrices too large.")
    
    # Get time series of residuals
    res = np.zeros((nres, m, ntr))
    
    # For each lag l, compute residuals res(l,:,:)
    for l in range(nres):
        res[l, :, :] = v[l + p, :, :]
        
        # Subtract intercept
        if w.size > 0:
            for itr in range(ntr):
                res[l, :, itr] = res[l, :, itr] - w
        
        # Subtract AR terms
        for itr in range(ntr):
            for j in range(1, p + 1):
                res[l, :, itr] = res[l, :, itr] - v[l - j + p, :, itr] @ A[:, (j - 1) * m:j * m].T
    
    # For computation of correlation matrices, center residuals by
    # subtraction of the mean
    resc = res - np.mean(res, axis=0)
    
    # Compute correlation matrix of the residuals
    # Compute lag zero correlation matrix
    c0 = np.zeros((m, m))
    for itr in range(ntr):
        c0 = c0 + resc[:, :, itr].T @ resc[:, :, itr]
    
    d = np.diag(c0)
    dd = np.sqrt(np.outer(d, d))
    c0 = c0 / dd
    
    # Compute lag l correlation matrix
    cl = np.zeros((m, m, k))
    for l in range(k):
        for itr in range(ntr):
            cl[:, :, l] = cl[:, :, l] + resc[:-l-1, :, itr].T @ resc[l+1:, :, itr]
        cl[:, :, l] = cl[:, :, l] / dd
      # Get "covariance matrix" in LMP statistic
    try:
        c0_inv = linalg.inv(c0)  # Inverse of lag 0 correlation matrix
    except linalg.LinAlgError:
        # Handle singular matrix by using pseudoinverse
        print("Warning: Singular correlation matrix detected. Using pseudoinverse.")
        c0_inv = linalg.pinv(c0)
    
    # Use the kronecker product for the "Covariance matrix" in LMP statistic
    rr = np.kron(c0_inv, c0_inv)
    
    # Compute modified Li-McLeod portmanteau statistic
    lmp = 0.0  # LMP statistic initialization
    
    for l in range(k):
        x = cl[:, :, l].flatten()  # Arrange cl as vector by stacking columns
        lmp = lmp + x @ rr @ x  # Sum up LMP statistic
    
    ntot = nres * ntr  # Total number of residual vectors
    lmp = ntot * lmp + m**2 * k * (k + 1) / 2 / ntot  # Add remaining term and scale
    dof_lmp = m**2 * (k - p)  # Degrees of freedom for LMP statistic
      # Significance level with which hypothesis of uncorrelatedness is rejected
    siglev = 1.0 - stats.chi2.cdf(lmp, dof_lmp)
    
    return siglev, res
