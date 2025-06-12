"""
DTF integration module for ARfit.

This module provides integration between the ARfit package and the DTF connectivity functions
in the FAST-IRES project. It replaces the placeholder AR model estimation in the DTF functions
with the proper ARfit implementation.
"""

import numpy as np
from scipy import linalg
from numba import njit
from typing import Tuple, Optional, Dict, Any, Union

from arfit.arfit import arfit

def prepare_ar_for_dtf(ts: np.ndarray, order: int, no_const: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate AR model parameters using ARfit for use with DTF functions.
    
    This function serves as a bridge between ARfit and the DTF connectivity functions.
    It handles the conversion between different parameter formats used in each module.
    
    Parameters
    ----------
    ts : np.ndarray
        Time series data, shape (n_samples, n_channels).
    order : int
        Model order to estimate.
    no_const : bool, optional (default: True)
        If True, fit a model without an intercept term (zero mean).
        If False, include an intercept term.
        
    Returns
    -------
    w : np.ndarray
        Noise covariance matrix, shape (n_channels, n_channels).
    A : np.ndarray
        AR coefficients matrix, shape (n_channels, n_channels * order).
    """
    # Use ARfit to estimate the model with specified order
    w_int, A_mat, C, _, _, _ = arfit(ts, order, order, no_const=no_const)
    
    # Return in the format expected by DTF functions (w is covariance matrix, A is coefficient matrix)
    return C, A_mat

@njit
def convert_ar_to_transfer(A: np.ndarray, nchan: int, order: int) -> np.ndarray:
    """
    Convert AR coefficients to transfer function format needed for DTF.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients, shape (n_channels, n_channels * order).
    nchan : int
        Number of channels.
    order : int
        Model order.
        
    Returns
    -------
    B : np.ndarray
        Transfer function coefficients, shape (n_channels, n_channels, order + 1).
    """
    B = np.zeros((nchan, nchan, order + 1), dtype=np.complex128)
    B[:, :, 0] = -np.eye(nchan)
    
    for i in range(nchan):
        for j in range(nchan):
            B[i, j, 1:order+1] = A[i, j::nchan]
            
    return B

def compute_dtf(ts: np.ndarray, low_freq: int, high_freq: int, order: int, fs: int = 400) -> np.ndarray:
    """
    Compute the Directed Transfer Function (DTF) using ARfit for model estimation.
    
    Parameters
    ----------
    ts : np.ndarray
        Time series data, shape (n_samples, n_channels).
    low_freq : int
        Lower bound of frequency range.
    high_freq : int
        Upper bound of frequency range.
    order : int
        Model order.
    fs : int, optional (default: 400)
        Sampling frequency in Hz.
        
    Returns
    -------
    gamma2 : np.ndarray
        DTF values, shape (n_channels, n_channels, n_frequencies).
    """
    try:
        # Try to use Numba-optimized version
        from .optimized_dtf import optimized_dtf
        print("Using Numba-optimized DTF calculation")
        return optimized_dtf(ts, low_freq, high_freq, order, fs)
    except ImportError:
        # Fallback to standard version
        print("Warning: Optimized DTF unavailable, using standard version")
        try:
            # Import here to avoid circular imports
            from Codes.DTF import DTF
            # Use the provided DTF function with our ARfit integration
            return DTF(ts, low_freq, high_freq, order, fs)
        except Exception as e:
            print(f"Error in DTF calculation: {e}")
            raise
