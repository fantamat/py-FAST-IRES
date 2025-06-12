"""
ARfit: Demonstration of modules in the ARfit package.

This script demonstrates the basic usage of the ARfit package for AR model
estimation, diagnostic checking, and eigenmode analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from .arsim import arsim
from .arfit import arfit
from .arres import arres
from .arconf import arconf
from .armode import armode
from .adjph import adjph
from .acf import acf

def run_demo():
    """
    Run a demonstration of the ARfit package functionality.
    
    This function demonstrates AR model fitting, residual analysis,
    and eigenmode decomposition using simulated data.
    """
    print("ARfit Demonstration")
    print("==================\n")
    
    print("ARfit is a collection of Python functions for the modeling of")
    print("multivariate time series with autoregressive (AR) models.")
    print("This demo illustrates the use of ARfit with a bivariate AR(2) process.\n")
    
    # Set up the AR(2) process parameters
    w = np.array([0.25, 0.1])  # Intercept vector
    
    A1 = np.array([[0.4, 1.2], [0.3, 0.7]])  # First lag coefficient matrix
    A2 = np.array([[0.35, -0.3], [-0.4, -0.5]])  # Second lag coefficient matrix
    
    A = np.hstack([A1, A2])  # Combined coefficient matrix
    
    C = np.array([[1.00, 0.50], [0.50, 1.50]])  # Noise covariance matrix
    
    print("Using the following AR(2) parameters:")
    print(f"Intercept w:\n{w}\n")
    print(f"AR coefficient matrix A1:\n{A1}\n")
    print(f"AR coefficient matrix A2:\n{A2}\n")
    print(f"Noise covariance matrix C:\n{C}\n")
    
    # Simulate time series
    n = 200  # Number of time steps
    ntr = 5  # Number of realizations
    
    print(f"Simulating {ntr} realizations with {n} observations each...")
    v = arsim(w, A, C, [n, ntr])
    
    print("Time series dimensions:", v.shape)
    print("Simulated time series has shape (time_steps, variables, realizations)\n")
    
    # Fit AR model with order selection
    print("Fitting AR model with order selection from pmin=1 to pmax=5...")
    pmin = 1
    pmax = 5
    
    west, Aest, Cest, SBC, FPE, th = arfit(v, pmin, pmax, selector='sbc', no_const=False)
    
    m = 2  # State space dimension 
    popt = Aest.shape[1] // m  # Estimated optimum order
    
    print(f"Estimated optimum model order: {popt}\n")
    
    print("Schwarz's Bayesian Criterion for orders 1 to 5:")
    print(SBC)
    
    print("\nAkaike's Final Prediction Error for orders 1 to 5:")
    print(FPE)
    
    # Check model adequacy
    print("\nChecking model adequacy (testing residuals for uncorrelatedness)...")
    siglev, res = arres(west, Aest, v)
    
    print(f"Significance level of Li-McLeod portmanteau test: {siglev:.4f}")
    if siglev > 0.05:
        print("The residuals appear to be uncorrelated (significance level > 0.05).")
    else:
        print("The residuals appear to be correlated (significance level <= 0.05).")
    
    # Compute confidence intervals
    print("\nComputing confidence intervals for AR parameters...")
    Aerr, werr = arconf(Aest, Cest, west, th)
    
    print("Estimated intercept vector with margins of error:")
    print(np.hstack([west, werr]))
    print("True intercept vector:")
    print(w.reshape(-1, 1))
    
    print("\nEstimated coefficient matrix:")
    print(Aest)
    print("With margins of error:")
    print(Aerr)
    print("True coefficients:")
    print(A)
    
    # Compute eigenmodes
    print("\nComputing eigenmodes of the fitted AR model...")
    S, Serr, per, tau, exctn, lambda_vals = armode(Aest, Cest, th)
    
    print("Estimated eigenmodes (columns of S):")
    print(S)
    print("\nMargins of error (Serr):")
    print(Serr)
    
    # Compute 'true' eigenmodes for comparison
    print("\nComputing 'true' eigenmodes from original parameters for comparison...")
    A1_true = np.vstack([A, np.hstack([np.eye(2), np.zeros((2, 2))])])
    lambda_true, S_true = linalg.eig(A1_true)
    S_true = adjph(S_true)
    S_true = S_true[2:4, :]
    
    print("True eigenmodes (note: order may differ from estimated):")
    print(S_true)
    
    print("\nEstimated periods and damping times:")
    print("Periods (per):")
    print(per)
    print("Damping times (tau):")
    print(tau)
    
    # Plot autocorrelation function if the data has multiple realizations
    print("\nPlotting autocorrelation function of the first component of residuals...")
    fig, ax, _ = acf(res[:, 0, 0])
    plt.tight_layout()
    
    return {
        'west': west,
        'Aest': Aest, 
        'Cest': Cest,
        'res': res,
        'siglev': siglev,
        'S': S,
        'Serr': Serr,
        'per': per,
        'tau': tau,
        'exctn': exctn,
        'lambda_vals': lambda_vals,
        'acf_fig': fig
    }

if __name__ == "__main__":
    results = run_demo()
    plt.show()
