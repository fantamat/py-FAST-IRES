"""
ARfit: Adjust phase of eigenvectors.

This module implements phase adjustment for complex eigenvectors.
"""

import numpy as np

def adjph(S):
    """
    Adjust phase of eigenvectors.
    
    Parameters
    ----------
    S : array_like
        Matrix whose columns are complex eigenvectors.
    
    Returns
    -------
    S : ndarray
        Matrix with adjusted complex eigenvectors.
    
    Notes
    -----
    This function multiplies a complex vector by a phase factor such that the 
    real part and the imaginary part of the vector are orthogonal and the norm 
    of the real part is greater than or equal to the norm of the imaginary part.
    
    This is required by ARMODE to normalize the eigenmodes of an AR model.
    """
    S = np.array(S)
    
    # Process each column of S
    for j in range(S.shape[1]):
        # If the imaginary part is not zero
        if np.any(np.imag(S[:, j]) != 0):
            # Get real and imaginary parts
            u = np.real(S[:, j])
            v = np.imag(S[:, j])
            
            # Compute phase factor
            alpha = np.dot(u, v) / np.dot(v, v)
            arg = 0.5 * np.arctan(2 * alpha / (1 - alpha**2))
            
            # Adjust eigenvector
            S[:, j] = S[:, j] * np.exp(1j * arg)
            
            # Check if the norm of the real part is less than the norm of the imaginary part
            u_new = np.real(S[:, j])
            v_new = np.imag(S[:, j])
            
            # If ||u|| < ||v||, multiply eigenvector by i
            if np.linalg.norm(u_new) < np.linalg.norm(v_new):
                S[:, j] = S[:, j] * 1j
    
    return S
