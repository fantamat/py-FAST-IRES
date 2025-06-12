"""
Example usage of the ARfit package for multivariate autoregressive model fitting.

This script demonstrates how to use the ARfit package to:
1. Generate synthetic AR process data
2. Fit an AR model
3. Test residuals
4. Analyze eigenmodes

This can be used as a template for your own analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
from arfit import arfit, arsim, arres, armode, arconf, acf

def generate_sample_data(model_order=2, length=1000, dimensions=2):
    """Generate sample data from an AR process."""
    # Define parameters for AR(p) process
    w = np.array([0.2, 0.1])  # Intercept
    
    # Create coefficient matrix
    A1 = np.array([[0.4, 0.3], [0.4, 0.5]])
    A2 = np.array([[0.2, 0.1], [0.1, 0.3]])
    A = np.hstack((A1, A2))  # Combined coefficients

    # Noise covariance
    C = np.array([[1.0, 0.5], [0.5, 1.2]])  
    
    # Simulate the process
    print(f"Simulating AR({model_order}) process...")
    v = arsim(w, A, C, length)
    
    return v, w, A, C

def fit_ar_model(data, p_min=1, p_max=10):
    """Fit an AR model with optimal order selection."""
    print(f"Fitting AR model with orders from {p_min} to {p_max}...")
    west, Aest, Cest, SBC, FPE, th = arfit(data, p_min, p_max)
    
    # Determine the optimal order
    optimal_order = Aest.shape[1] // data.shape[1]
    print(f"Optimal order selected: {optimal_order}")
    
    return west, Aest, Cest, th, SBC, FPE

def check_residuals(west, Aest, data):
    """Check residuals for whiteness (uncorrelated residuals)."""
    print("Checking residual autocorrelations...")
    siglev, res = arres(west, Aest, data)
    
    print(f"Significance level: {siglev}")
    if siglev > 0.05:
        print("Residuals appear uncorrelated (test passed)")
    else:
        print("Residuals appear correlated (possible model misspecification)")
    
    # Plot ACF of first residual
    fig, ax, rho = acf(res[:, 0, 0], caption="Residual Autocorrelation")
    plt.tight_layout()
    
    return res, siglev

def analyze_eigenmodes(Aest, Cest, th):
    """Analyze the eigenmodes of the fitted AR model."""
    print("Computing eigenmodes...")
    S, Serr, per, tau, exctn, lambda_vals = armode(Aest, Cest, th)
    
    print("Oscillation periods:")
    print(per[0])
    print("Damping times:")
    print(tau[0])
    print("Excitations (normalized):")
    print(exctn)
    
    # Create indices for all modes
    indices = np.arange(len(exctn))
    
    # Filter out purely relaxatory modes (infinite period)
    oscillatory_indices = [i for i in indices if not np.isinf(per[0, i])]
    
    print("\nOscillatory modes:")
    for i in oscillatory_indices:
        print(f"Mode {i+1}: Period = {per[0, i]:.2f}, Damping = {tau[0, i]:.2f}, Excitation = {exctn[i]:.2f}")
    
    return S, per, tau, exctn, lambda_vals

def main():
    """Main function demonstrating the ARfit workflow."""
    # Generate sample data
    data, true_w, true_A, true_C = generate_sample_data()
    
    # Fit AR model
    west, Aest, Cest, th, SBC, FPE = fit_ar_model(data)
    
    # Show estimated parameters
    print(f"True intercept: {true_w}")
    print(f"Estimated intercept: {west.flatten()}")
    print(f"True coefficients:\n{true_A}")
    print(f"Estimated coefficients:\n{Aest}")
    
    # Compute confidence intervals for parameters
    Aerr, werr = arconf(Aest, Cest, west, th)
    print("\nConfidence intervals for intercept:")
    for i in range(len(west)):
        print(f"  w[{i}] = {west[i, 0]:.4f} ± {werr[i, 0]:.4f}")
    
    # Check residuals
    res, siglev = check_residuals(west, Aest, data)
    
    # Analyze eigenmodes
    S, per, tau, exctn, lambda_vals = analyze_eigenmodes(Aest, Cest, th)
    
    # Plot the eigenvalue locations in the complex plane
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(lambda_vals), np.imag(lambda_vals), c=exctn, cmap='viridis', 
                s=100, alpha=0.7, edgecolors='k')
    
    # Add unit circle
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.3)
    
    # Add labels
    for i in range(len(lambda_vals)):
        plt.annotate(f"{i+1}", 
                     (np.real(lambda_vals[i]), np.imag(lambda_vals[i])),
                     xytext=(10, 10), textcoords='offset points')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Eigenvalues in the Complex Plane')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.colorbar(label='Excitation (normalized)')
    plt.axis('equal')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()
