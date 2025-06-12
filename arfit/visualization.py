"""
Visualization tools for AR model analysis.

This module provides visualization functions for AR models, including:
- Plotting eigenvalues in the complex plane
- Frequency response visualization
- Multiple-trial data visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional, Dict, Any, Union

def plot_eigenvalues(eigvals: np.ndarray, title: str = "AR Model Eigenvalues") -> plt.Figure:
    """
    Plot eigenvalues of an AR model in the complex plane.
    
    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues of the AR model.
    title : str, optional
        Title for the plot. Default is "AR Model Eigenvalues".
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
    ax.add_artist(circle)
    
    # Plot eigenvalues
    ax.scatter(np.real(eigvals), np.imag(eigvals), c='b', marker='x', s=50)
    
    # Add labels
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title(title)
    
    # Set axis limits to be slightly larger than the unit circle
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    
    # Add grid
    ax.grid(True, linestyle=':')
    
    # Set the aspect ratio to be equal
    ax.set_aspect('equal')
    
    return fig

def plot_frequency_response(A: np.ndarray, C: np.ndarray, fs: float = 1.0, 
                           nfreqs: int = 512, db_scale: bool = False) -> plt.Figure:
    """
    Plot the frequency response of an AR model.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficient matrices, concatenated as [A1, A2, ..., Ap].
    C : np.ndarray
        Noise covariance matrix.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    nfreqs : int, optional
        Number of frequency points. Default is 512.
    db_scale : bool, optional
        If True, use decibel scale for magnitude. Default is False.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    m = C.shape[0]  # Number of channels
    p = A.shape[1] // m  # Model order
    
    # Construct array of AR coefficients for scipy.signal.freqz
    # NOTE: The format for freqz is different from ARfit's A
    ar_coeffs = np.zeros((m, m, p + 1))
    ar_coeffs[:, :, 0] = np.eye(m)
    
    for i in range(p):
        ar_coeffs[:, :, i + 1] = -A[:, i*m:(i+1)*m]
    
    # Compute frequency response for each channel pair
    w = np.linspace(0, np.pi, nfreqs)  # Normalized frequency
    freq_hz = w * fs / (2 * np.pi)  # Convert to Hz
    
    # Create subplots for each channel pair
    fig, axes = plt.subplots(m, m, figsize=(4*m, 3*m))
    if m == 1:
        axes = np.array([[axes]])
    
    for i in range(m):
        for j in range(m):
            h_freq = np.zeros(nfreqs, dtype=complex)
            
            # Compute transfer function
            for k in range(nfreqs):
                # Compute H(z) = 1/A(z) at z = e^(jw)
                z_inv_powers = np.array([np.exp(-1j * w[k] * l) for l in range(p + 1)])
                a_z = np.zeros((m, m), dtype=complex)
                
                for l in range(p + 1):
                    a_z += ar_coeffs[:, :, l] * z_inv_powers[l]
                
                # H(z) = A(z)^(-1)
                try:
                    h_z = np.linalg.inv(a_z)
                    h_freq[k] = h_z[i, j]
                except np.linalg.LinAlgError:
                    h_freq[k] = np.nan
            
            ax = axes[i, j]
            
            # Plot magnitude
            if db_scale:
                ax.plot(freq_hz, 20 * np.log10(np.abs(h_freq)))
                ax.set_ylabel('Magnitude (dB)')
            else:
                ax.plot(freq_hz, np.abs(h_freq))
                ax.set_ylabel('Magnitude')
            
            # Set title and labels
            ax.set_title(f'From Channel {j+1} to Channel {i+1}')
            ax.set_xlabel('Frequency (Hz)')
            ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_ar_residuals(v: np.ndarray, residuals: np.ndarray, 
                     max_lags: int = 20) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot time series data and residuals from an AR model fit.
    
    Parameters
    ----------
    v : np.ndarray
        Original time series data.
    residuals : np.ndarray
        Residuals from the AR model fit.
    max_lags : int, optional
        Maximum number of lags to include in the autocorrelation plot. Default is 20.
        
    Returns
    -------
    fig_ts : matplotlib.figure.Figure
        Figure containing the time series and residuals plot.
    fig_acf : matplotlib.figure.Figure
        Figure containing the autocorrelation plots.
    """
    if v.ndim > 2:
        # If multiple trials, average across trials
        v_plot = np.mean(v, axis=2)
        residuals_plot = np.mean(residuals, axis=2)
    else:
        v_plot = v
        residuals_plot = residuals
    
    m = v_plot.shape[1]  # Number of channels
    t = np.arange(v_plot.shape[0])
    
    # Time series and residuals plot
    fig_ts, axes = plt.subplots(m, 2, figsize=(14, 3 * m))
    
    if m == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(m):
        # Original time series
        ax_ts = axes[i, 0]
        ax_ts.plot(t, v_plot[:, i])
        ax_ts.set_title(f'Channel {i+1} Time Series')
        ax_ts.set_xlabel('Time (samples)')
        ax_ts.set_ylabel('Amplitude')
        ax_ts.grid(True)
        
        # Residuals
        ax_res = axes[i, 1]
        ax_res.plot(t[-len(residuals_plot):], residuals_plot[:, i], 'r')
        ax_res.set_title(f'Channel {i+1} Residuals')
        ax_res.set_xlabel('Time (samples)')
        ax_res.set_ylabel('Amplitude')
        ax_res.grid(True)
    
    plt.tight_layout()
    
    # Autocorrelation plot
    fig_acf, axes_acf = plt.subplots(m, 2, figsize=(14, 3 * m))
    
    if m == 1:
        axes_acf = axes_acf.reshape(1, -1)
    
    for i in range(m):
        # Original time series autocorrelation
        ax_acf_ts = axes_acf[i, 0]
        acf_ts = acf(v_plot[:, i], max_lags)
        ax_acf_ts.stem(np.arange(len(acf_ts)), acf_ts, use_line_collection=True)
        ax_acf_ts.set_title(f'Channel {i+1} Time Series ACF')
        ax_acf_ts.set_xlabel('Lag')
        ax_acf_ts.set_ylabel('Autocorrelation')
        ax_acf_ts.grid(True)
        
        # Add confidence bounds
        conf_level = 1.96 / np.sqrt(len(v_plot))
        ax_acf_ts.axhline(y=conf_level, linestyle='--', color='r')
        ax_acf_ts.axhline(y=-conf_level, linestyle='--', color='r')
        
        # Residuals autocorrelation
        ax_acf_res = axes_acf[i, 1]
        acf_res = acf(residuals_plot[:, i], max_lags)
        ax_acf_res.stem(np.arange(len(acf_res)), acf_res, use_line_collection=True)
        ax_acf_res.set_title(f'Channel {i+1} Residuals ACF')
        ax_acf_res.set_xlabel('Lag')
        ax_acf_res.set_ylabel('Autocorrelation')
        ax_acf_res.grid(True)
        
        # Add confidence bounds
        conf_level = 1.96 / np.sqrt(len(residuals_plot))
        ax_acf_res.axhline(y=conf_level, linestyle='--', color='r')
        ax_acf_res.axhline(y=-conf_level, linestyle='--', color='r')
    
    plt.tight_layout()
    
    return fig_ts, fig_acf

def acf(x, max_lags):
    """
    Compute the autocorrelation function.
    
    Parameters
    ----------
    x : np.ndarray
        Input time series.
    max_lags : int
        Maximum number of lags to compute.
        
    Returns
    -------
    acf_values : np.ndarray
        Autocorrelation values for lags 0 to max_lags.
    """
    # Ensure the input is mean-centered
    x = x - np.mean(x)
    
    # Compute the autocorrelation using np.correlate
    result = np.correlate(x, x, mode='full')
    
    # Extract only the positive lags
    n = len(x)
    acf_values = result[n-1:n+max_lags] / result[n-1]
    
    return acf_values
