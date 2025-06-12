"""
This module provides a convenient interface for using the arfit package in the FAST-IRES project.

It reexports the main functions from the arfit package for easy access and provides
additional functionality specific to the FAST-IRES project.
"""

import numpy as np
from scipy import stats, linalg
import matplotlib.pyplot as plt

# Import arfit functions
from arfit import arfit, arsim, arres, armode, arconf

def estimate_ar_model(data, p_min=1, p_max=10, no_const=False, selector='sbc'):
    """
    Estimate an AR model from the given data with automatic order selection.
    
    Parameters
    ----------
    data : array_like
        Time series data. If data is a matrix, columns represent variables.
        If data has three dimensions, the third dimension corresponds to trials.
    p_min : int, optional
        Minimum model order to consider. Default is 1.
    p_max : int, optional
        Maximum model order to consider. Default is 10.
    no_const : bool, optional
        If True, fit a model without an intercept term. Default is False.
    selector : str, optional
        Criterion for order selection. Either 'sbc' or 'fpe'. Default is 'sbc'.
        
    Returns
    -------
    dict
        A dictionary containing the estimated model parameters:
        - 'order': The selected order of the AR model.
        - 'w': The intercept vector.
        - 'A': The AR coefficient matrices.
        - 'C': The noise covariance matrix.
        - 'SBC': Schwarz's Bayesian Criterion for each tested order.
        - 'FPE': Akaike's Final Prediction Error for each tested order.
        - 'th': Information needed for confidence intervals.
    """
    w, A, C, SBC, FPE, th = arfit(data, p_min, p_max, selector=selector, no_const=no_const)
    
    # Determine the selected order
    if selector == 'sbc':
        selected_idx = np.argmin(SBC)
    else:  # 'fpe'
        selected_idx = np.argmin(FPE)
    
    selected_order = p_min + selected_idx
    
    return {
        'order': selected_order,
        'w': w,
        'A': A,
        'C': C,
        'SBC': SBC,
        'FPE': FPE,
        'th': th
    }

def check_ar_model_residuals(w, A, data, max_lag=None):
    """
    Check if the residuals of an AR model are uncorrelated.
    
    Parameters
    ----------
    w : array_like
        Intercept vector of the AR model.
    A : array_like
        AR coefficient matrices.
    data : array_like
        Original time series data used to fit the model.
    max_lag : int, optional
        Maximum lag of residual correlation matrices to consider.
        If None, an appropriate value is selected automatically.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'significant': Boolean indicating if residuals are significantly correlated.
        - 'p_value': The p-value of the test for uncorrelated residuals.
        - 'residuals': The time series of residuals.
        - 'statistic': The value of the Li-McLeod portmanteau statistic.
        - 'dof': Degrees of freedom for the statistic.
    """
    siglev, res, lmp, dof_lmp = arres(w, A, data, k=max_lag)
    
    return {
        'significant': siglev <= 0.05,
        'p_value': siglev,
        'residuals': res,
        'statistic': lmp,
        'dof': dof_lmp
    }

def analyze_ar_eigenmodes(A, C, th):
    """
    Analyze the eigenmodes of an AR model to identify oscillatory components.
    
    Parameters
    ----------
    A : array_like
        AR coefficient matrices.
    C : array_like
        Noise covariance matrix.
    th : array_like
        Information about the fit, as returned by arfit.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'modes': Matrix whose columns contain the eigenmodes.
        - 'errors': Margins of error for the eigenmodes.
        - 'periods': Oscillation periods and their margins of error.
        - 'damping_times': Damping times and their margins of error.
        - 'excitations': Normalized excitations of the eigenmodes.
        - 'eigenvalues': Eigenvalues of the AR model.
    """
    S, Serr, per, tau, exctn, lambda_vals = armode(A, C, th)
    
    # Sort modes by excitation
    idx = np.argsort(exctn)[::-1]  # Sort in descending order
    
    # Reorder all results
    S = S[:, idx]
    Serr = Serr[:, idx]
    per = per[:, idx]
    tau = tau[:, idx]
    exctn = exctn[idx]
    lambda_vals = lambda_vals[idx]
    
    return {
        'modes': S,
        'errors': Serr,
        'periods': per,
        'damping_times': tau,
        'excitations': exctn,
        'eigenvalues': lambda_vals
    }

def get_dominant_frequencies(modes_result, sampling_rate=1.0, min_excitation=0.05):
    """
    Extract the dominant frequencies from AR model eigenmodes.
    
    Parameters
    ----------
    modes_result : dict
        Dictionary returned by analyze_ar_eigenmodes.
    sampling_rate : float, optional
        Sampling rate of the original data in Hz. Default is 1.0.
    min_excitation : float, optional
        Minimum excitation threshold for modes to be considered. Default is 0.05.
    
    Returns
    -------
    dict
        A dictionary with:
        - 'frequencies': The dominant frequencies in Hz.
        - 'excitations': The excitations of the dominant modes.
        - 'damping_times': The damping times of the dominant modes.
        - 'indices': The indices of the dominant modes.
    """
    periods = modes_result['periods'][0]  # First row contains the periods
    exctn = modes_result['excitations']
    tau = modes_result['damping_times'][0]  # First row contains the damping times
    
    # Convert periods to frequencies (in Hz if sampling_rate is in Hz)
    # Handle infinite periods (purely relaxatory modes)
    frequencies = np.zeros_like(periods)
    for i, p in enumerate(periods):
        if np.isinf(p) or p == 0:
            frequencies[i] = 0  # Zero frequency for purely relaxatory modes
        else:
            frequencies[i] = sampling_rate / p
    
    # Find dominant oscillatory modes (non-zero frequency and significant excitation)
    indices = np.where((frequencies > 0) & (exctn >= min_excitation))[0]
    
    return {
        'frequencies': frequencies[indices],
        'excitations': exctn[indices],
        'damping_times': tau[indices],
        'indices': indices
    }
