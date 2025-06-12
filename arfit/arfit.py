"""
ARfit: A Python package for the estimation of parameters and eigenmodes of multivariate autoregressive models.

This module implements the main arfit function that performs stepwise least squares estimation of multivariate AR models.
"""

import numpy as np
from scipy import linalg
from .arqr import arqr
from .arord import arord

# Try to import Numba optimizations for acceleration
try:
    from .numba_optimized import (
        qr_factorization, compute_sbc_fpe, compute_eigendecomposition, 
        compute_periods_damping, compute_order_criteria, optimized_residuals_test
    )
    HAS_NUMBA = True
    print("ARfit: Using Numba-optimized functions for acceleration")
except ImportError as e:
    HAS_NUMBA = False
    print(f"ARfit: Numba not available, using standard numpy/scipy implementations. Error: {e}")

def arfit(v, pmin, pmax, selector='sbc', no_const=True):
    """
    Stepwise least squares estimation of multivariate AR model.
    
    Parameters
    ----------
    v : array_like
        Time series data. If v is a matrix, each column v[:,j] is assumed to contain
        a univariate time series. If v is a 3D array, each v[:,:,k] is treated as a
        distinct realization (trial) of the underlying multivariate time series.
    pmin : int
        Minimum model order to consider.
    pmax : int
        Maximum model order to consider.
    selector : str, optional (default: 'sbc')
        Criterion used for order selection. Either 'sbc' for Schwarz's Bayesian
        Criterion or 'fpe' for the logarithm of Akaike's Final Prediction Error.
    no_const : bool, optional (default: True)
        If True, fit a model without an intercept term (zero mean).
        If False, include an intercept term.
    
    Returns
    -------
    w : ndarray
        Estimated intercept vector of size (m,1). If no_const is True, w is a zero vector.
    A : ndarray
        Estimated AR coefficient matrices, concatenated as [A1, A2, ..., Ap]
        where Ai is the coefficient matrix for lag i.
    C : ndarray
        Estimated noise covariance matrix.
    sbc : ndarray
        Schwarz's Bayesian Criterion for each model order from pmin to pmax.
    fpe : ndarray
        Logarithm of Akaike's Final Prediction Error for each model order from pmin to pmax.
    th : ndarray
        Matrix containing information needed for computation of confidence intervals.
        The first row contains degrees of freedom, and the remaining rows contain
        the inverse of U'*U, where U is used in the asymptotic covariance matrix.
    
    Notes
    -----
    This function estimates an m-variate AR model of order p:
    
        v(k,:) = w' + A1*v(k-1,:)' + ... + Ap*v(k-p,:)' + noise
    
    The noise is assumed to be Gaussian with zero mean and covariance matrix C.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    
    Schneider, T., and A. Neumaier, 2001: Algorithm 808: ARfit - A Matlab package for
    the estimation of parameters and eigenmodes of multivariate autoregressive models.
    ACM Trans. Math. Software, 27, 58-65.
    """
    # Get dimensions
    v = np.array(v)
    if v.ndim == 1:
        v = v.reshape(-1, 1)  # Convert to column vector
    
    if v.ndim == 2:
        n, m = v.shape  # n time steps, m variables
        ntr = 1
        v = v.reshape(n, m, 1)  # Add singleton dimension for trials
    else:  # 3D array
        n, m, ntr = v.shape  # n time steps, m variables, ntr trials
    
    if pmin != int(pmin) or pmax != int(pmax):
        raise ValueError("Order must be integer.")
    
    if pmax < pmin:
        raise ValueError("PMAX must be greater than or equal to PMIN.")
    
    # Set mcor based on no_const
    mcor = 0 if no_const else 1
    
    # Compute number of equations and parameters
    ne = ntr * (n - pmax)  # Number of block equations of size m
    npmax = m * pmax + mcor  # Maximum number of parameter vectors of length m
    
    if ne <= npmax:
        raise ValueError("Time series too short.")
    
    # Compute QR factorization for model of order pmax
    R, scale = arqr(v, pmax, mcor)
      # Compute order selection criteria for models of order pmin:pmax
    sbc, fpe, _, _ = arord(R, m, mcor, ne, pmin, pmax)
    
    # Get index of order that minimizes the order selection criterion
    if selector == 'sbc':
        iopt = np.argmin(sbc)
    elif selector == 'fpe':
        iopt = np.argmin(fpe)
    else:
        raise ValueError("Invalid selector. Use 'sbc' or 'fpe'.")
    
    # Select order of model
    popt = pmin + iopt  # Estimated optimum order
    np_opt = m * popt + mcor  # Number of parameter vectors of length m
    
    # Decompose R for the optimal model order popt
    R11 = R[:np_opt, :np_opt]
    R12 = R[:np_opt, npmax:npmax+m]
    R22 = R[np_opt:npmax+m, npmax:npmax+m]
    
    # Get augmented parameter matrix Aaug=[w A] if mcor=1 and Aaug=A if mcor=0
    if np_opt > 0:
        if mcor == 1:
            # Improve condition of R11 by re-scaling first column
            con = np.max(scale[1:npmax+m]) / scale[0]
            R11_scaled = R11.copy()
            R11_scaled[:, 0] = R11[:, 0] * con
            Aaug = linalg.solve(R11_scaled, R12).T
            
            # Return coefficient matrix A and intercept vector w separately
            w = (Aaug[:, 0] * con).reshape(-1, 1)
            A = Aaug[:, 1:np_opt]
        else:
            # No intercept vector
            Aaug = linalg.solve(R11, R12).T
            w = np.zeros((m, 1))
            A = Aaug
    else:
        # No parameters have been estimated
        w = np.zeros((m, 1))
        A = np.array([])
    
    # Return covariance matrix
    dof = ne - np_opt  # Number of block degrees of freedom
    C = (R22.T @ R22) / dof  # Bias-corrected estimate of covariance matrix
    
    # For later computation of confidence intervals return in th:
    # (i) The inverse of U=R11'*R11, which appears in the asymptotic covariance matrix
    # (ii) The number of degrees of freedom of the residual covariance matrix
    invR11 = linalg.inv(R11)
    if mcor == 1:
        # Undo condition improving scaling
        invR11[0, :] = invR11[0, :] * con
    
    Uinv = invR11 @ invR11.T
    th = np.vstack((np.array([dof] + [0] * (Uinv.shape[1] - 1)), Uinv))
    
    return w, A, C, sbc, fpe, th
