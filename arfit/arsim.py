"""
ARfit: Simulation of AR processes.

This module implements the simulation of AR processes.
"""

import numpy as np
from scipy import linalg

def arsim(w, A, C, n_ntr, ndisc=None):
    """
    Simulation of AR process.
    
    Parameters
    ----------
    w : array_like
        Intercept vector of the AR model.
    A : array_like
        AR coefficient matrices, concatenated as [A1, A2, ..., Ap].
    C : array_like
        Noise covariance matrix.
    n_ntr : int or tuple
        If int, simulates n_ntr time steps of the AR process.
        If tuple of length 2, simulates n_ntr[0] time steps with n_ntr[1] realizations.
    ndisc : int, optional
        Number of initial time steps to discard to avoid spin-up effects.
        If None, 1000 time steps are discarded.
    
    Returns
    -------
    v : ndarray
        Simulated time series. If n_ntr is a tuple, v is a 3D array where
        v[:,:,itr] is the itr-th realization.
    
    Notes
    -----
    This function simulates an AR(p) process:
    
        v(k,:)' = w' + A1*v(k-1,:)' + ... + Ap*v(k-p,:)' + eta(k,:)'
    
    where eta(k,:) are independent Gaussian noise vectors with mean zero and
    covariance matrix C.
    
    The p vectors of initial values for the simulation are taken to be equal to
    the mean value of the process.
    
    References
    ----------
    Neumaier, A., and T. Schneider, 2001: Estimation of parameters and eigenmodes of
    multivariate autoregressive models. ACM Trans. Math. Software, 27, 27-57.
    """
    # Convert inputs to numpy arrays
    w = np.array(w).flatten()
    A = np.array(A)
    C = np.array(C)
    
    # Get dimensions
    m = C.shape[0]  # Dimension of state vectors
    p = A.shape[1] // m  # Order of process
    
    # Parse n_ntr
    if isinstance(n_ntr, (list, tuple)):
        n = n_ntr[0]  # Number of time steps
        ntr = n_ntr[1]  # Number of realizations
    else:
        n = n_ntr  # Number of time steps
        ntr = 1  # Default: one realization
    
    if p != int(p):
        raise ValueError("Bad arguments.")
    
    if len(w) != m or w.ndim > 1:
        raise ValueError("Dimensions of arguments are mutually incompatible.")
    
    # Convert w to row vector
    w = w.reshape(1, -1)
    
    # Check whether specified model is stable
    A1 = np.vstack([A, np.hstack([np.eye((p-1)*m), np.zeros(((p-1)*m, m))])])
    lambda_vals = linalg.eigvals(A1)
    if np.any(np.abs(lambda_vals) > 1):
        print("Warning: The specified AR model is unstable.")
    
    # Discard the first ndisc time steps; if ndisc is not given as input
    # argument, use default
    if ndisc is None:
        ndisc = 10**3
    
    # Compute Cholesky factor of covariance matrix C
    try:
        R = linalg.cholesky(C).T  # R is upper triangular
    except linalg.LinAlgError:
        raise ValueError("Covariance matrix not positive definite.")
    
    # Get ntr realizations of ndisc+n independent Gaussian
    # pseudo-random vectors with covariance matrix C=R'*R
    randvec = np.zeros((ndisc+n, m, ntr))
    for itr in range(ntr):
        randvec[:, :, itr] = np.random.randn(ndisc+n, m) @ R
    
    # Add intercept vector to random vectors
    for itr in range(ntr):
        randvec[:, :, itr] = randvec[:, :, itr] + w
    
    # Get transpose of system matrix A (use transpose in simulation because
    # we want to obtain the states as row vectors)
    AT = A.T
    
    # Take the p initial values of the simulation to equal the process mean,
    # which is calculated from the parameters A and w
    if np.any(w):
        # Process has nonzero mean: mval = inv(B)*w' where
        # B = eye(m) - A1 - ... - Ap
        # Assemble B
        B = np.eye(m)
        for j in range(p):
            B = B - A[:, j*m:(j+1)*m]
        
        # Get mean value of process
        try:
            mval = linalg.solve(B.T, w.T).T
        except:
            # Use pseudoinverse if matrix is singular
            mval = w @ np.linalg.pinv(B)
        
        # Initialize with mean value
        x = np.tile(mval, (p, 1))
    else:
        # Process has zero mean
        x = np.zeros((p, m))
    
    # Initialize state vectors
    u = np.zeros((ndisc+n+p, m, ntr))
    for itr in range(ntr):
        u[:p, :, itr] = x
    
    # Simulate ntr realizations of n+ndisc time steps. In order to be
    # able to make use of vectorization capabilities, the cases p=1
    # and p>1 must be treated separately.
    if p == 1:
        for itr in range(ntr):
            for k in range(1, ndisc+n+1):
                x[0, :] = u[k-1, :, itr] @ AT
                u[k, :, itr] = x + randvec[k-1, :, itr]
    else:
        for itr in range(ntr):
            for k in range(p, ndisc+n+p):
                x = np.zeros((p, m))
                for j in range(1, p+1):
                    x[j-1, :] = u[k-j, :, itr] @ AT[(j-1)*m:j*m, :]
                u[k, :, itr] = np.sum(x, axis=0) + randvec[k-p, :, itr]
    
    # Return only the last n simulated state vectors
    if ntr == 1:
        return u[ndisc+p:ndisc+n+p, :, 0]  # Return 2D array for a single realization
    else:
        return u[ndisc+p:ndisc+n+p, :, :]  # Return 3D array for multiple realizations
