"""
ARfit: Autocorrelation function plotting.

This module implements plotting of sample autocorrelation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def acf(x, k=25, caption="ACF"):
    """
    Plot of sample autocorrelation function.
    
    Parameters
    ----------
    x : array_like
        Univariate time series data.
    k : int, optional
        Maximum lag for which to compute the autocorrelation.
        Default is 25.
    caption : str, optional
        Title for the plot. Default is "ACF".
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    rho : ndarray
        Autocorrelation values.
    
    Notes
    -----
    This function plots the sample autocorrelation function of a univariate time series.
    The approximate 95% confidence limits of the autocorrelation function of an IID process
    of the same length are also displayed. Sample autocorrelations lying outside the 95%
    confidence intervals of an IID process are marked with asterisks.
    """
    x = np.asarray(x).flatten()  # Ensure x is a 1D array
    
    n = len(x)
    
    # Compute autocorrelation sequence
    cor = signal.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    cor = cor[n-1:] / cor[n-1]  # Normalize and take only positive lags
    rho = cor[:k+1]  # Autocorrelation function up to lag k
    
    # Approximate 95% confidence limits for IID process of the same length
    bound = np.ones_like(rho) * 1.96 / np.sqrt(n)
    
    # Find lags within and outside approximate 95% confidence
    # intervals; start with lag 0
    within_indices = []
    outside_indices = []
    
    for i in range(1, k+1):  # Skip lag 0
        if np.abs(rho[i]) > bound[i]:  # Point outside confidence intervals
            outside_indices.append(i)
        else:  # Point within confidence intervals
            within_indices.append(i)
    
    # Plot ACF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    ax.plot(range(k+1), rho, 'b-', label='Autocorrelation')
    ax.plot(range(k+1), bound, 'r-.', label='95% Confidence Limits')
    ax.plot(range(k+1), -bound, 'r-.', label='_nolegend_')
    
    # Mark points outside confidence intervals with asterisks
    if outside_indices:
        ax.plot(outside_indices, rho[outside_indices], 'r*', label='Outside 95% CI', markersize=10)
    
    # Mark points within confidence intervals with circles
    if within_indices:
        ax.plot(within_indices, rho[within_indices], 'bo', label='Within 95% CI', markersize=5)
    
    # Add lag 0 point
    ax.plot(0, rho[0], 'bo', markersize=5)
    
    ax.set_xlim(0, k)
    ax.set_ylim(-1, 1)
    ax.set_title(caption)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax, rho
