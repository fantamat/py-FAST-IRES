"""
Numba-accelerated versions of core ARfit computational functions.

This module provides accelerated implementations of the most computationally
intensive functions in the ARfit package using Numba JIT compilation.
"""

import numpy as np
from scipy import linalg
from numba import jit, njit
import warnings

@njit
def qr_factorization(K_padded):
    """
    Numba-accelerated QR factorization for AR model fitting.
    
    Parameters
    ----------
    K_padded : ndarray
        Pre-processed input matrix for QR factorization.
    
    Returns
    -------
    R : ndarray
        Upper triangular matrix from QR factorization.
    """
    # Using numpy's QR implementation since Numba doesn't support scipy.linalg.qr directly
    Q, R = np.linalg.qr(K_padded)
    return R

@njit
def compute_sbc_fpe(R, m, mcor, ne, p):
    """
    Numba-accelerated computation of SBC and FPE criteria.
    
    Parameters
    ----------
    R : ndarray
        Upper triangular factor in the QR factorization of the AR model.
    m : int
        Dimension of state vectors (number of variables).
    mcor : int
        Flag indicating whether an intercept vector is fitted.
    ne : int
        Number of block equations of size m used in the estimation.
    p : int
        Model order.
    
    Returns
    -------
    sbc_val : float
        Schwarz's Bayesian Criterion value.
    fpe_val : float
        Logarithm of Akaike's Final Prediction Error.
    """
    n = ne - p            # Effective number of observations
    np_var = m * m * p    # Number of AR parameters
    
    if mcor == 1:
        np_var = np_var + m  # Add parameters for the intercept
    
    # Get residual covariance from R
    Rm = R[-m:, -m:]
    
    # Log determinant of residual covariance
    logdp = 2 * np.log(np.abs(np.diag(Rm))).sum()
    
    # Schwarz's Bayesian Criterion
    sbc_val = logdp + np.log(n) * np_var / n
    
    # Logarithm of Akaike's Final Prediction Error
    fpe_val = logdp + np.log((n + np_var) / (n - np_var))
    
    return sbc_val, fpe_val

@njit
def compute_eigendecomposition(A_companion, m, p):
    """
    Numba-accelerated eigendecomposition of AR companion matrix.
    
    Parameters
    ----------
    A_companion : ndarray
        Companion matrix form of AR coefficients.
    m : int
        Dimension of state vectors (number of variables).
    p : int
        Model order.
    
    Returns
    -------
    eigvals : ndarray
        Eigenvalues of the companion matrix.
    eigvecs : ndarray
        Eigenvectors of the companion matrix.
    """
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(A_companion)
    
    # Sort eigenvalues by magnitude (descending)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

@njit
def compute_periods_damping(eigvals, fs=1.0):
    """
    Numba-accelerated computation of oscillation periods and damping times.
    
    Parameters
    ----------
    eigvals : ndarray
        Eigenvalues of the AR companion matrix.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    
    Returns
    -------
    periods : ndarray
        Oscillation periods for each eigenvalue.
    damping : ndarray
        Damping times for each eigenvalue.
    """
    n = len(eigvals)
    periods = np.zeros(n, dtype=np.float64)
    damping = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if np.abs(eigvals[i]) > 0:
            if np.abs(np.imag(eigvals[i])) > 1e-10:
                # Complex eigenvalue - compute period and damping
                periods[i] = 2 * np.pi / np.abs(np.angle(eigvals[i])) / fs
                damping[i] = -1.0 / np.log(np.abs(eigvals[i])) / fs
            else:
                # Real eigenvalue - no oscillation
                periods[i] = 0.0
                damping[i] = -1.0 / np.log(np.abs(eigvals[i])) / fs
        else:
            # Zero eigenvalue
            periods[i] = 0.0
            damping[i] = 0.0
    
    return periods, damping

@jit(nopython=False, forceobj=True)  # We need object mode for linalg operations
def compute_order_criteria(R, m, mcor, ne, pmin, pmax):
    """
    Numba-accelerated computation of order selection criteria.
    
    Parameters
    ----------
    R : ndarray
        Upper triangular factor in the QR factorization of the AR model.
    m : int
        Dimension of state vectors (number of variables).
    mcor : int
        Flag indicating whether an intercept vector is fitted.
    ne : int
        Number of block equations of size m used in the estimation.
    pmin : int
        Minimum model order to consider.
    pmax : int
        Maximum model order to consider.
    
    Returns
    -------
    sbc : ndarray
        Schwarz's Bayesian Criterion for each order.
    fpe : ndarray
        Logarithm of Akaike's Final Prediction Error for each order.
    """
    imax = pmax - pmin + 1  # Maximum index of output vectors
    
    # Initialize output vectors
    sbc = np.zeros(imax)  # Schwarz's Bayesian Criterion
    fpe = np.zeros(imax)  # Log of Akaike's Final Prediction Error
    logdp = np.zeros(imax)  # Determinant of (scaled) covariance matrix
    np_params = np.zeros(imax, dtype=np.int32)  # Number of parameter vectors of length m
    np_params[imax - 1] = m * pmax + mcor
    
    # Get lower right triangle R22 of R
    start_idx = int(np_params[imax - 1])
    end_idx = start_idx + m
    R22 = R[start_idx:end_idx, start_idx:end_idx]
    
    # From R22, get inverse of residual cross-product matrix for model of order pmax
    try:
        invR22 = linalg.inv(R22)
        Mp = invR22 @ invR22.T
    except linalg.LinAlgError:
        # Use pseudoinverse if inversion fails
        invR22 = linalg.pinv(R22)
        Mp = invR22 @ invR22.T
    
    # For order selection, get determinant of residual cross-product matrix
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
        
        # Criteria calculation
        n = ne - p  # Effective number of observations
        nparam = m * m * p + m * mcor  # Number of parameters
        
        # Schwarz's Bayesian Criterion
        sbc[i] = logdp[i] / m + np.log(n) * nparam / n
        
        # Logarithm of Akaike's Final Prediction Error
        fpe[i] = logdp[i] / m + np.log((n + nparam) / (n - nparam))
        
        i -= 1
    
    return sbc, fpe

@jit(nopython=False, forceobj=True)
def optimized_residuals_test(v, w, A, C, test_lag=20):
    """
    Numba-accelerated test for residual autocorrelations.
    
    Parameters
    ----------
    v : ndarray
        Time series data.
    w : ndarray
        Estimated intercept.
    A : ndarray
        Estimated AR coefficients.
    C : ndarray
        Estimated noise covariance.
    test_lag : int, optional
        Maximum lag for testing autocorrelations. Default is 20.
    
    Returns
    -------
    h : bool
        Test result. True if residuals appear uncorrelated, False otherwise.
    sig : float
        Significance level at which the null hypothesis (uncorrelated residuals) can be rejected.
    res : ndarray
        Residuals from the AR model.
    """
    # Get dimensions
    if v.ndim == 3:  # Multiple trials
        n, m, ntrial = v.shape
    else:
        n, m = v.shape
        ntrial = 1
        v = v.reshape(n, m, 1)
    
    # Get model order from A
    p = A.shape[1] // m
    
    # Preallocate residuals matrix
    ne = n - p
    res = np.zeros((ne, m, ntrial))
    
    # Compute residuals for each trial
    for trial in range(ntrial):
        # Get residuals: res[k,j,trial] = v[k+p,j,trial] - w[j] - sum_i sum_l A[j,i,l]*v[k+p-l,i,trial]
        for k in range(ne):
            # Start with the intercept term
            for j in range(m):
                res[k, j, trial] = v[k+p, j, trial] - w[j]
                
                # Subtract AR terms
                for l in range(1, p+1):
                    for i in range(m):
                        res[k, j, trial] -= A[j, (l-1)*m + i] * v[k+p-l, i, trial]
    
    # Pool all trials for autocorrelation check
    res_pooled = res.reshape(-1, m, order='F').T  # Reshape for efficient computation
    
    # Make residual vectors zero-mean
    for i in range(m):
        res_pooled[i, :] = res_pooled[i, :] - np.mean(res_pooled[i, :])
    
    # Compute sample autocorrelations up to lag test_lag
    r = np.zeros((m, m, test_lag+1))
    n_pooled = res_pooled.shape[1]
    
    for k in range(test_lag + 1):
        for i in range(m):
            for j in range(m):
                # Sum of products for lag k
                r[i, j, k] = np.sum(res_pooled[i, :n_pooled-k] * res_pooled[j, k:]) / n_pooled
    
    # Normalize by lag 0 autocorrelations to get correlation coefficients
    r0_inv = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if r[i, i, 0] > 0 and r[j, j, 0] > 0:
                r0_inv[i, j] = 1.0 / np.sqrt(r[i, i, 0] * r[j, j, 0])
    
    for k in range(test_lag + 1):
        r[:, :, k] = r[:, :, k] * r0_inv
    
    # Portmanteau test for significance of autocorrelations
    c0 = r[:, :, 0]  # Lag 0 correlation (should be identity matrix if properly normalized)
    
    # Compute test statistic
    Q = 0.0
    for k in range(1, test_lag + 1):
        c = r[:, :, k]  # Correlation at lag k
        try:
            c0_inv = linalg.inv(c0)
            Q += np.trace(c.T @ c0_inv @ c @ c0_inv) * n_pooled / (n_pooled - k)
        except linalg.LinAlgError:
            # If c0 is singular, use pseudoinverse
            c0_inv = linalg.pinv(c0)
            Q += np.trace(c.T @ c0_inv @ c @ c0_inv) * n_pooled / (n_pooled - k)
    
    # Significance level (p-value)
    df = test_lag * m * m  # Degrees of freedom
    sig = 1.0  # Default: autocorrelations not significant
    
    # Check if we have enough samples for a reliable test
    if n_pooled > test_lag + m:
        from scipy.stats import chi2
        sig = 1.0 - chi2.cdf(Q, df)
    
    # Test result: h = True if residuals appear uncorrelated
    h = sig > 0.05
    
    return h, sig, res.reshape(ne, m, ntrial)
