"""
Test script for the ARfit package.

This script provides a simple test of the ARfit package functionality.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the package is in the path
sys.path.append(os.path.abspath('.'))

try:
    from arfit import arfit, arsim, arres, armode
    print("Successfully imported arfit modules!")
except Exception as e:
    print(f"Error importing arfit: {e}")
    sys.exit(1)

def test_arfit():
    """Test the ARfit package with a simple AR(2) model."""
    print("Testing ARfit package...")
    
    # Define parameters for AR(2) process
    w = np.array([0.2, 0.1])  # Intercept
    A1 = np.array([[0.4, 0.3], [0.4, 0.5]])
    A2 = np.array([[0.2, 0.1], [0.1, 0.3]])
    A = np.hstack((A1, A2))  # Combined coefficients
    C = np.array([[1.0, 0.5], [0.5, 1.2]])  # Noise covariance
    
    # Simulate time series
    n = 1000  # Time steps
    print("Simulating AR(2) process...")
    v = arsim(w, A, C, n)
    
    # Fit AR model with order selection
    print("Fitting AR model with orders from 1 to 5...")
    pmin = 1
    pmax = 5
    west, Aest, Cest, SBC, FPE, th = arfit(v, pmin, pmax)
    
    # Display results
    print(f"Optimal order selected: {Aest.shape[1] // 2}")
    print(f"True intercept: {w}")
    print(f"Estimated intercept: {west.flatten()}")
    print(f"True coefficients:\n{A}")
    print(f"Estimated coefficients:\n{Aest}")
    
    # Check residuals
    print("Checking residual autocorrelations...")
    siglev, res = arres(west, Aest, v)
    print(f"Significance level: {siglev}")
    
    if siglev > 0.05:
        print("Residuals appear uncorrelated (test passed)")
    else:
        print("Residuals appear correlated (possible model misspecification)")
    
    # Compute eigenmodes
    print("Computing eigenmodes...")
    S, Serr, per, tau, exctn, lambda_vals = armode(Aest, Cest, th)
    
    # Display eigenmode results
    print("Oscillation periods:")
    print(per[0])
    print("Damping times:")
    print(tau[0])
    print("Excitations (normalized):")
    print(exctn)
    
    print("ARfit test completed successfully!")
    return True

if __name__ == "__main__":
    test_arfit()
