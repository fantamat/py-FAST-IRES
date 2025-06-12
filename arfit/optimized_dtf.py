"""
Optimized DTF calculations for FAST-IRES using the ARfit package.

This module provides Numba-accelerated DTF (Directed Transfer Function) calculations 
that are compatible with the ARfit package.
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional

@njit
def compute_dtf_from_ar_params(A: np.ndarray, nchan: int, p: int, 
                              frequencies: np.ndarray, fs: float = 400.0) -> np.ndarray:
    """
    Compute DTF from AR model parameters.
    
    This function separates the AR model parameter estimation from the DTF computation,
    allowing the DTF computation to be accelerated with Numba.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients, shape (n_channels, n_channels * model_order)
    nchan : int
        Number of channels
    p : int
        Model order
    frequencies : np.ndarray
        Array of frequencies to compute DTF for
    fs : float, optional
        Sampling frequency in Hz. Default is 400.0.
        
    Returns
    -------
    gamma2 : np.ndarray
        DTF values, shape (n_channels, n_channels, n_frequencies)
    """
    nfre = len(frequencies)
    dt = 1.0 / fs
    
    # Rearrange the format of the MVAR matrix
    B = np.zeros((nchan, nchan, p + 1), dtype=np.complex128)
    B[:, :, 0] = -np.eye(nchan)
    
    for i in range(nchan):
        for j in range(nchan):
            B[i, j, 1:p+1] = A[i, j::nchan]
    
    # Calculate DTF for all frequencies
    gamma2 = np.zeros((nchan, nchan, nfre), dtype=np.float64)
    
    for fre_idx in range(nfre):
        fre = frequencies[fre_idx]
        Bf = np.zeros((nchan, nchan), dtype=np.complex128)
        
        for ind_l in range(p + 1):
            Bf = Bf + B[:, :, ind_l] * np.exp(-1j * 2 * np.pi * fre * dt * ind_l)
        
        H = np.zeros((nchan, nchan), dtype=np.complex128)
        
        try:
            H = np.linalg.inv(Bf)
        except:
            # If matrix is singular, use small perturbation
            Bf = Bf + np.eye(nchan) * 1e-10
            H = np.linalg.inv(Bf)
        
        # Calculate gamma2 (DTF)
        for i in range(nchan):
            for j in range(nchan):
                gamma2[i, j, fre_idx] = np.abs(H[i, j]) ** 2 / np.sum(np.abs(H[i, :]) ** 2)
    
    return gamma2

def optimized_dtf(ts: np.ndarray, low_freq: int, high_freq: int, p: int, fs: float = 400.0) -> np.ndarray:
    """
    Compute DTF with separated AR parameter estimation and DTF calculation.
    
    This function uses ARfit for the AR model parameter estimation and then calls
    a Numba-accelerated function for the DTF computation.
    
    Parameters
    ----------
    ts : np.ndarray
        Time series data, shape (n_samples, n_channels)
    low_freq : int
        Lower bound of frequency range
    high_freq : int
        Upper bound of frequency range
    p : int
        Model order
    fs : float, optional
        Sampling frequency in Hz. Default is 400.0.
        
    Returns
    -------
    gamma2 : np.ndarray
        DTF values, shape (n_channels, n_channels, n_frequencies)
    """
    # Import here to avoid circular imports
    from arfit.dtf_integration import prepare_ar_for_dtf
    
    # Estimate AR model parameters
    w, A = prepare_ar_for_dtf(ts, p)
    
    # Define frequencies
    frequencies = np.arange(low_freq, high_freq + 1)
    nchan = ts.shape[1]
    
    # Compute DTF using Numba-accelerated function
    gamma2 = compute_dtf_from_ar_params(A, nchan, p, frequencies, fs)
    
    return gamma2
