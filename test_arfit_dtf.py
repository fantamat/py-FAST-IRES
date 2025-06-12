"""
Test script for ARfit integration with DTF connectivity functions.

This script verifies that the ARfit implementation works correctly with the
DTF connectivity functions from the FAST-IRES project.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the directories are in the path
sys.path.append(os.path.abspath('.'))

try:    # Import ARfit modules
    from arfit import arfit, arsim, armode
    from arfit.dtf_integration import prepare_ar_for_dtf, compute_dtf
    
    # Add the current directory to the path to make Codes imports work
    sys.path.append(os.path.abspath('.'))
    
    print("Successfully imported all required modules!")
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_arfit_dtf_integration():
    """Test the integration of ARfit with DTF connectivity functions."""
    print("Testing ARfit-DTF integration...")
    
    # Define parameters for AR(2) process with known connectivity pattern
    m = 4  # Number of channels
    p = 2  # Model order
    
    # Create intercept and coefficients
    w = np.zeros(m)
    
    # Create AR coefficients with specific connectivity pattern:
    # Channel 1 -> Channel 2, Channel 3 -> Channel 4
    A1 = np.zeros((m, m))
    A1[1, 0] = 0.5  # Channel 1 influences Channel 2
    A1[3, 2] = 0.6  # Channel 3 influences Channel 4
    
    A2 = np.zeros((m, m))
    A2[1, 0] = 0.3  # Channel 1 influences Channel 2
    A2[3, 2] = 0.2  # Channel 3 influences Channel 4
    
    # Combined AR coefficients
    A = np.hstack((A1, A2))
    
    # Noise covariance
    C = np.eye(m)
    
    # Simulation parameters
    n = 2000  # Time steps
    print(f"Simulating {m}-channel AR({p}) process with {n} samples...")
    
    # Simulate the time series
    v = arsim(w, A, C, n)
    
    # Compute DTF using ARfit
    print("Computing DTF using ARfit integration...")
    fs = 100  # Sampling frequency (Hz)
    low_freq = 1
    high_freq = 45    # Method 1: Using our integration function
    gamma2_integrated = compute_dtf(v, low_freq, high_freq, p, fs)
    
    # Method 2: Using our optimized DTF function
    from arfit.optimized_dtf import optimized_dtf
    gamma2_optimized = optimized_dtf(v, low_freq, high_freq, p, fs)
    
    # For comparison, let's also try direct parameter extraction and DTF calculation
    w, A = prepare_ar_for_dtf(v, p)
    from arfit.optimized_dtf import compute_dtf_from_ar_params
    frequencies = np.arange(low_freq, high_freq + 1)
    gamma2_direct = compute_dtf_from_ar_params(A, m, p, frequencies, fs)
      # Compare the results
    diff1 = np.max(np.abs(gamma2_integrated - gamma2_optimized))
    diff2 = np.max(np.abs(gamma2_optimized - gamma2_direct))
    print(f"Maximum difference between integration and optimized: {diff1}")
    print(f"Maximum difference between optimized and direct: {diff2}")
      # Plot the DTF values
    plot_dtf_connectivity(gamma2_integrated, m, low_freq, high_freq)
    
    # Verify connectivity patterns
    verify_connectivity(gamma2_integrated, m)
    
    return gamma2_integrated, gamma2_optimized, gamma2_direct
    
    return gamma2_integrated

def plot_dtf_connectivity(gamma2, m, low_freq, high_freq):
    """Plot DTF connectivity results."""
    frequencies = np.linspace(low_freq, high_freq, gamma2.shape[2])
    
    fig, axes = plt.subplots(m, m, figsize=(12, 10))
    fig.suptitle('DTF Connectivity')
    
    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            ax.plot(frequencies, gamma2[i, j, :])
            ax.set_ylim([0, 1])
            
            # Add labels only to edges
            if i == m-1:
                ax.set_xlabel('Frequency (Hz)')
            if j == 0:
                ax.set_ylabel('DTF')
                
            # Set title only for top row
            if i == 0:
                ax.set_title(f'From {j+1} to {i+1}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('dtf_connectivity.png')
    plt.close()

def verify_connectivity(gamma2, m):
    """Verify that the connectivity patterns are correct."""
    # The expected pattern is:
    # 1 -> 2, 3 -> 4
    # Calculate mean DTF across frequencies
    mean_dtf = np.mean(gamma2, axis=2)
    
    print("\nConnectivity strength (averaged across frequencies):")
    for i in range(m):
        for j in range(m):
            print(f"From {j+1} to {i+1}: {mean_dtf[i, j]:.4f}")
    
    # Check if the expected connections are the strongest
    expected_connections = [(1, 0), (3, 2)]  # (target, source) in zero-based indexing
    
    for target, source in expected_connections:
        # Check if the connection from source to target is strong
        source_idx = np.argsort(mean_dtf[target, :])[::-1]
        if source_idx[0] == source:
            print(f"✓ Verified: Channel {source+1} -> Channel {target+1} is the strongest connection for Channel {target+1}")
        else:
            print(f"✗ Unexpected: Channel {source+1} -> Channel {target+1} is NOT the strongest connection")

if __name__ == "__main__":
    test_arfit_dtf_integration()
