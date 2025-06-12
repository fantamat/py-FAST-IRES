"""
ARfit: Eigendecomposition of AR model.

This module implements eigendecomposition of AR models and computation of 
oscillation periods and damping times.
"""

import numpy as np
from scipy import linalg
from .tquant import tquant
from .adjph import adjph

def armode(A, C, th):
    """
    Eigendecomposition of AR model.
    
    Parameters
    ----------
    A : array_like
        AR coefficient matrices, concatenated as [A1, A2, ..., Ap].
    C : array_like
        Noise covariance matrix.
    th : array_like
        Information about the fit, as returned by arfit.
    
    Returns
    -------
    S : ndarray
        Matrix whose columns contain the estimated eigenmodes of the AR model.
    Serr : ndarray
        Margins of error for the components of the estimated eigenmodes.
    per : ndarray
        Oscillation periods of the eigenmodes.
        per[0, k] is the estimated period of eigenmode S[:, k].
        per[1, k] is the margin of error for this period.
    tau : ndarray
        Damping times of the eigenmodes.
        tau[0, k] is the estimated damping time of eigenmode S[:, k].
        tau[1, k] is the margin of error for this damping time.
    exctn : ndarray
        Excitation of the eigenmodes, normalized so that the sum equals one.
    lambda_vals : ndarray
        Eigenvalues of the AR model.
    
    Notes
    -----
    For a purely relaxatory eigenmode, the period is infinite (inf).
    For an oscillatory eigenmode, the periods are finite.
    
    The excitation of an eigenmode measures its dynamical importance.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    """
    A = np.array(A)
    C = np.array(C)
    th = np.array(th)
    
    # Set confidence coefficient
    ccoeff = 0.95
    m = C.shape[0]  # Dimension of state space
    p = A.shape[1] // m  # Order of process
    
    if p <= 0:
        raise ValueError("Order must be greater than 0.")
    
    # Assemble coefficient matrix of equivalent AR(1) model
    A1 = np.vstack([A, np.hstack([np.eye((p-1)*m), np.zeros(((p-1)*m, m))])])
    
    # Eigenvalues and eigenvectors of coefficient matrix of equivalent AR(1) model
    lambda_vals, BigS = linalg.eig(A1)
    
    # Warning if the estimated model is unstable
    if np.any(np.abs(lambda_vals) > 1):
        print("Warning: The estimated AR model is unstable.\n\tSome excitations may be negative.")
    
    # Fix phase of eigenvectors such that the real part and the imaginary part
    # of each vector are orthogonal
    BigS = adjph(BigS)
    
    # Return only last m components of each eigenvector
    S = BigS[(p-1)*m:p*m, :]
    
    # Compute inverse of BigS for later use
    BigS_inv = linalg.inv(BigS)
    
    # Recover the matrix Uinv that appears in the asymptotic covariance
    # matrix of the least squares estimator (Uinv is output of AR)
    if th.shape[1] == m*p + 1:
        # The intercept vector has been fitted by AR; in computing
        # confidence intervals for the eigenmodes, this vector is
        # irrelevant. The first row and first column in Uinv,
        # corresponding to elements of the intercept vector, are not needed.
        Uinv = th[1:, 1:]
    elif th.shape[1] == m*p:
        # No intercept vector has been fitted
        Uinv = th[1:, :]
    else:
        raise ValueError("Input arguments of ARMODE must be output of ARFIT.")
    
    # Number of degrees of freedom
    dof = th[0, 0]
    
    # Quantile of t distribution for given confidence coefficient and dof
    t = tquant(dof, 0.5 + ccoeff/2)
    
    # Asymptotic covariance matrix of estimator of coefficient matrix A
    Sigma_A = np.kron(Uinv, C)
    
    # Noise covariance matrix of system of relaxators and oscillators
    CovDcpld = BigS_inv[:, :m] @ C @ BigS_inv[:, :m].T
    
    # For each eigenmode j: compute the period per, the damping time tau,
    # and the excitation exctn; also get the margins of error for per and tau
    mp = m * p  # Total number of modes
    per = np.zeros((2, mp))  # First row: periods, second row: margins of error
    tau = np.zeros((2, mp))  # First row: damping times, second row: margins of error
    exctn = np.zeros(mp)  # Excitations
    
    for j in range(mp):
        a = np.real(lambda_vals[j])  # Real part of eigenvalue j
        b = np.imag(lambda_vals[j])  # Imaginary part of eigenvalue j
        abs_lambda_sq = a**2 + b**2  # Squared absolute value of eigenvalue j
        abs_lambda = np.sqrt(abs_lambda_sq)  # Absolute value of eigenvalue j
        
        # Compute excitation: excitation is proportional to the diagonal element
        # of the covariance matrix of the system of relaxators and oscillators
        exctn[j] = np.real(CovDcpld[j, j])
        
        # Guard against round-off making the excitation a very small
        # negative number
        if exctn[j] < 0 and np.abs(exctn[j]) < 1e-10:
            exctn[j] = 0
        
        # Compute damping time tau(1,j) from eigenvalue
        if abs_lambda > 0.0:
            tau[0, j] = -1.0 / np.log(abs_lambda)
        else:
            tau[0, j] = 0.0
        
        # Compute derivatives of the absolute value of eigenvalue j with respect
        # to the elements of the coefficient matrix A ("dot" denotes the
        # derivative)
        if abs_lambda > 0.0:
            # Derivative of modulus with respect to real part
            abs_lambda_dot_a = a / abs_lambda
            
            # Derivative of modulus with respect to imag part
            abs_lambda_dot_b = b / abs_lambda
            
            # Derivatives of real and imag parts of eigenvalue with respect to
            # elements of the coefficient matrix A
            dot_a = np.real(BigS_inv[j, :m])
            dot_b = np.imag(BigS_inv[j, :m])
              # Get derivative of log(abs_lambda) with respect to A
            phi = (1.0 / abs_lambda) * (abs_lambda_dot_a * dot_a + abs_lambda_dot_b * dot_b)
            
            # Convert to derivative of tau = -1/log(abs_lambda) with respect to A
            phi = tau[0, j]**2 / abs_lambda * (a * dot_a + b * dot_b)
            
            # Margin of error for damping time tau - ensure phi is the right shape
            # Create a vector of the correct size to match Sigma_A dimensions
            phi_vec = np.zeros(m * m * p)
            for i in range(min(len(phi), len(phi_vec))):
                phi_vec[i] = phi[i]
                
            # Calculate the margin of error
            tau[1, j] = t * np.sqrt(phi_vec @ Sigma_A @ phi_vec)
            
            # Period of eigenmode j and margin of error for period
            if b == 0 and a >= 0:  # Purely real, nonnegative eigenvalue
                per[0, j] = float('inf')
                per[1, j] = 0.0
            elif b == 0 and a < 0:  # Purely real, negative eigenvalue
                per[0, j] = 2.0
                per[1, j] = 0.0
            else:  # Complex eigenvalue
                per[0, j] = 2.0 * np.pi / np.abs(np.arctan2(b, a))
                  # Derivative of period with respect to parameters in A
                phi = per[0, j]**2 / (2.0 * np.pi * abs_lambda_sq) * (b * dot_a - a * dot_b)
                
                # Margin of error for period - ensure phi is the right shape
                # Create a vector of the correct size to match Sigma_A dimensions
                phi_vec = np.zeros(m * m * p)
                for i in range(min(len(phi), len(phi_vec))):
                    phi_vec[i] = phi[i]
                    
                # Calculate the margin of error
                per[1, j] = t * np.sqrt(phi_vec @ Sigma_A @ phi_vec)
    
    # Give the excitation as 'relative importance' that sums to one
    # Handle the case where sum of excitation is zero or negative
    if np.sum(exctn) > 0:
        exctn = exctn / np.sum(exctn)
    else:
        exctn = np.ones_like(exctn) / len(exctn)  # Uniform distribution if invalid
    
    # Compute confidence intervals for eigenmodes
    # -------------------------------------------
    # Shorthands for matrix products
    XX = np.real(BigS).T @ np.real(BigS)
    YY = np.imag(BigS).T @ np.imag(BigS)
    XY = np.real(BigS).T @ np.imag(BigS)
    
    # Need confidence intervals only for last m rows of BigS
    row1 = (p-1)*m  # First row for which confidence interval is needed (zero-indexed)
    
    Serr = np.zeros_like(S)
    
    for k in range(mp):  # Loop over columns of S
        # Get orthogonal transformation matrix T that block-diagonalizes
        # the covariance matrix of the real and imaginary parts of eigenmodes
        if np.imag(lambda_vals[k]) == 0.0:            # Real eigenvalue => no need to compute the transformation T
            # => Sigma_S = The part of Sigma_AR corresponding to rows row1:row1+m-1
            # Determine the covariance matrix of ReS[:,k]
            phi_re_S = np.real(BigS_inv[k, :m])
            
            # Create appropriate sized matrix for the error calculation
            # We need to handle the dimensions properly
            err_variance = np.zeros(m)
            for i in range(m):
                # Create a vector with a 1 in the i-th position
                test_vec = np.zeros(m * p * m)
                indices = np.where(phi_re_S != 0)[0]
                for j in indices:
                    if j < len(test_vec):
                        test_vec[j] = phi_re_S[j]
                
                # Calculate variance for this component
                if np.any(test_vec != 0):
                    err_variance[i] = test_vec @ Sigma_A @ test_vec
            
            # Assign the appropriate portion to the error margins
            Serr[:, k] = t * np.sqrt(err_variance)
        else:
            # Complex eigenvalue => need to compute T
            # Get index of complex conjugate eigenvalue
            k_cc = -1
            for kk in range(mp):
                if np.abs(np.imag(lambda_vals[kk]) + np.imag(lambda_vals[k])) < 1e-10 and kk != k:
                    k_cc = kk
                    break
            
            if k_cc < 0:
                raise ValueError("Could not find complex conjugate eigenvalue.")
            
            # Skip if already processed
            if k_cc < k:
                continue
            
            # Get inverse eigenvalue problems for the pair of complex conjugate
            # eigenvalues that generate the eigenmodes k and k_cc
            phi_k = BigS_inv[k, :m]
            phi_k_cc = BigS_inv[k_cc, :m]
            
            # Get derivative of ReS[:,k] with respect to A
            phi_re_S = 0.5 * (phi_k + phi_k_cc)
            
            # Get derivative of ImS[:,k] with respect to A
            phi_im_S = 0.5j * (phi_k - phi_k_cc)
              # Get covariance matrix of ReS[:,k] and ImS[:,k]
            # We need to handle the dimensions properly for both real and imaginary parts
            err_variance_re = np.zeros(m)
            err_variance_im = np.zeros(m)
            
            for i in range(m):
                # For real part errors
                test_vec_re = np.zeros(m * p * m)
                indices_re = np.where(np.real(phi_re_S) != 0)[0]
                for j in indices_re:
                    if j < len(test_vec_re):
                        test_vec_re[j] = np.real(phi_re_S[j])
                
                if np.any(test_vec_re != 0):
                    err_variance_re[i] = test_vec_re @ Sigma_A @ test_vec_re
                
                # For imaginary part errors
                test_vec_im = np.zeros(m * p * m)
                indices_im = np.where(np.imag(phi_im_S) != 0)[0]
                for j in indices_im:
                    if j < len(test_vec_im):
                        test_vec_im[j] = np.imag(phi_im_S[j])
                
                if np.any(test_vec_im != 0):
                    err_variance_im[i] = test_vec_im @ Sigma_A @ test_vec_im
            
            # Margins of error for ReS[:,k] and ImS[:,k]
            Serr_re = t * np.sqrt(err_variance_re)
            Serr_im = t * np.sqrt(err_variance_im)
            
            # Assign margins of error
            Serr[:, k] = Serr_re + 1j * Serr_im
            if k_cc != k:
                Serr[:, k_cc] = Serr_re - 1j * Serr_im
    
    return S, Serr, per, tau, exctn, lambda_vals
