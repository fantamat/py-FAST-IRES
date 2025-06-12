"""
ARfit: Order selection criteria for AR models.

This module implements the evaluation of criteria for selecting the order of an AR model.
"""

import numpy as np
from scipy import linalg

def arord(R, m, mcor, ne, pmin, pmax):
    """
    Evaluates criteria for selecting the order of an AR model.
    
    Parameters
    ----------
    R : ndarray
        Upper triangular factor in the QR factorization of the AR model.
    m : int
        Dimension of state vectors (number of variables).
    mcor : int
        Flag indicating whether an intercept vector is being fitted (1) or not (0).
    ne : int
        Number of block equations of size m used in the estimation.
    pmin : int
        Minimum model order to consider.
    pmax : int
        Maximum model order to consider.
    
    Returns
    -------
    sbc : ndarray
        Schwarz's Bayesian Criterion for each model order from pmin to pmax.
    fpe : ndarray
        Logarithm of Akaike's Final Prediction Error for each model order from pmin to pmax.
    logdp : ndarray
        Logarithm of the determinant of the (scaled) covariance matrix for each order.
    np_params : ndarray
        Number of parameter vectors for each order.
    
    Notes
    -----
    The returned values of the order selection criteria are approximate in that
    in evaluating a selection criterion for an AR model of order p < pmax,
    pmax-p initial values of the given time series are ignored.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    """
    imax = pmax - pmin + 1  # Maximum index of output vectors
    
    # Initialize output vectors
    sbc = np.zeros(imax)  # Schwarz's Bayesian Criterion
    fpe = np.zeros(imax)  # Log of Akaike's Final Prediction Error
    logdp = np.zeros(imax)  # Determinant of (scaled) covariance matrix
    np_params = np.zeros(imax, dtype=int)  # Number of parameter vectors of length m
    np_params[imax - 1] = m * pmax + mcor
      # Get lower right triangle R22 of R
    start_idx = int(np_params[imax - 1])
    end_idx = start_idx + m
    R22 = R[start_idx:end_idx, start_idx:end_idx]
    
    # From R22, get inverse of residual cross-product matrix for model of order pmax
    invR22 = linalg.inv(R22)
    Mp = invR22 @ invR22.T
    
    # For order selection, get determinant of residual cross-product matrix
    # logdp = log det(residual cross-product matrix)
    logdp[imax - 1] = 2.0 * np.log(np.abs(np.prod(np.diag(R22))))
    
    # Compute approximate order selection criteria for models of order pmin:pmax
    i = imax - 1
    for p in range(pmax, pmin - 1, -1):
        np_params[i] = m * p + mcor  # Number of parameter vectors of length m
        
        if p < pmax:
            # Downdate determinant of residual cross-product matrix
            # Rp: Part of R to be added to Cholesky factor of covariance matrix
            Rp = R[np_params[i]:np_params[i] + m, np_params[imax - 1]:np_params[imax - 1] + m]
            
            # Get Mp, the downdated inverse of the residual cross-product
            # matrix, using the Woodbury formula
            I_m = np.eye(m)
            try:
                L = linalg.cholesky(I_m + Rp @ Mp @ Rp.T, lower=True)
                N = linalg.solve(L, Rp @ Mp)
                Mp = Mp - N.T @ N
                
                # Get downdated logarithm of determinant
                logdp[i] = logdp[i + 1] + 2.0 * np.log(np.abs(np.prod(np.diag(L))))
            except linalg.LinAlgError:
                # If Cholesky factorization fails, use a direct approach
                inv_matrix = linalg.inv(I_m + Rp @ Mp @ Rp.T)
                Mp = Mp - Mp @ Rp.T @ inv_matrix @ Rp @ Mp
                logdp[i] = logdp[i + 1] + np.log(np.abs(linalg.det(I_m + Rp @ Mp @ Rp.T)))
        
        # Schwarz's Bayesian Criterion
        sbc[i] = logdp[i] / m - np.log(ne) * (ne - np_params[i]) / ne
        
        # Logarithm of Akaike's Final Prediction Error
        fpe[i] = logdp[i] / m - np.log(ne * (ne - np_params[i]) / (ne + np_params[i]))
        
        # Modified Schwarz criterion (MSC) - commented out in original
        # msc(i) = logdp(i)/m - (log(ne) - 2.5) * (1 - 2.5*np(i)/(ne-np(i)))
        
        i -= 1  # Go to next lower order
    
    return sbc, fpe, logdp, np_params
