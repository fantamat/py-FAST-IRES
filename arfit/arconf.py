"""
ARfit: Confidence intervals for AR coefficients.

This module implements the computation of confidence intervals for AR coefficients.
"""

import numpy as np
from .tquant import tquant

def arconf(A, C, w_or_th, th=None):
    """
    Confidence intervals for AR coefficients.
    
    Parameters
    ----------
    A : array_like
        Estimated AR coefficient matrices, concatenated as [A1, A2, ..., Ap].
    C : array_like
        Estimated noise covariance matrix.
    w_or_th : array_like
        If th is not None, this is the intercept vector w.
        Otherwise, this is the matrix th containing information needed for confidence intervals.
    th : array_like, optional
        Matrix containing information needed for confidence intervals.
    
    Returns
    -------
    Aerr : ndarray
        Margins of error for the elements of the coefficient matrix A.
    werr : ndarray, optional
        Margins of error for the components of the intercept vector w.
        Only returned if an intercept vector was fitted.
    
    Notes
    -----
    For an AR(p) model that has been fitted with ARFIT, this function computes
    the margins of error Aerr and werr such that (A +/- Aerr) and (w +/- werr)
    are approximate 95% confidence intervals for the elements of the coefficient
    matrix A and for the components of the intercept vector w.
    
    The confidence intervals are based on Student's t distribution, which for small
    samples yields only approximate confidence intervals. Inferences drawn from small
    samples must therefore be interpreted cautiously.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    """
    # Set confidence coefficient
    ccoeff = 0.95
    
    # Convert inputs to numpy arrays
    A = np.array(A)
    C = np.array(C)
    
    m = C.shape[0]  # Dimension of state space
    p = A.shape[1] // m  # Order of model
    
    # Check if intercept vector was fitted
    if th is None:
        # No intercept vector has been fitted
        Aaug = A
        th = w_or_th
        w = None
        np_param = m * p  # Number of parameter vectors of size m
    else:
        # Intercept vector has been fitted
        w = w_or_th
        Aaug = np.hstack([w, A])
        np_param = m * p + 1  # Number of parameter vectors of size m
    
    # Number of degrees of freedom for residual covariance matrix
    dof = th[0, 0]
    
    # Quantile of t distribution for given confidence coefficient and dof
    t = tquant(dof, 0.5 + ccoeff/2)
    
    # Get matrix Uinv that appears in the covariance matrix of the least squares estimator
    Uinv = th[1:, :]
    
    # Compute approximate confidence intervals for elements of Aaug
    Aaug_err = np.zeros((m, np_param))
    for j in range(m):
        for k in range(np_param):
            Aaug_err[j, k] = t * np.sqrt(Uinv[k, k] * C[j, j])
    
    if w is None:
        # No intercept vector has been fitted
        return Aaug_err
    else:
        # An intercept vector has been fitted => return margins of error
        # for intercept vector and for AR coefficients separately
        werr = Aaug_err[:, 0].reshape(-1, 1)
        Aerr = Aaug_err[:, 1:]
        return Aerr, werr
