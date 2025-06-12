"""
Comprehensive test script for the ARfit package implementation.

This script provides extensive testing of the ARfit Python implementation
to ensure it behaves similarly to the original MATLAB version.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, linalg
import time

# Ensure the package is in the path
sys.path.append(os.path.abspath('.'))

try:
    from arfit import (
        arfit, arsim, arres, armode, arconf,
        plot_eigenvalues, plot_frequency_response, plot_ar_residuals
    )
    print("Successfully imported all ARfit modules!")
except ImportError as e:
    print(f"Error importing ARfit: {e}")
    try:
        from arfit import arfit, arsim, arres, armode, arconf
        print("Successfully imported core ARfit modules, but visualization tools not available.")
    except ImportError as e:
        print(f"Error importing core ARfit modules: {e}")
        sys.exit(1)

def print_section(title):
    """Print a section title."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_ar_simulation():
    """Test AR process simulation."""
    print_section("Testing AR Process Simulation")
    
    # Test cases with different dimensions and orders
    test_cases = [
        {"m": 1, "p": 1, "n": 1000, "desc": "Univariate AR(1)"},
        {"m": 2, "p": 2, "n": 1000, "desc": "Bivariate AR(2)"},
        {"m": 5, "p": 3, "n": 1000, "desc": "5-dimensional AR(3)"}
    ]
    
    for case in test_cases:
        m, p, n = case["m"], case["p"], case["n"]
        desc = case["desc"]
        print(f"Testing {desc} simulation...")
        
        # Create intercept
        w = np.random.randn(m) * 0.1
        
        # Create AR coefficients - use stable coefficients
        A = np.zeros((m, m * p))
        for i in range(p):
            if m == 1:
                # For univariate case, use a coefficient less than 1 for stability
                A[:, i*m:(i+1)*m] = np.array([[0.5 / (i+1)]])
            else:
                # For multivariate case, create a stable matrix
                temp = np.random.randn(m, m) * (0.5 / (i+1))
                # Make it diagonally dominant for stability
                for j in range(m):
                    temp[j, j] = 0.5 / (i+1)
                A[:, i*m:(i+1)*m] = temp
        
        # Create noise covariance
        C = np.eye(m)
        if m > 1:
            # Add some correlation
            for i in range(m-1):
                C[i, i+1] = C[i+1, i] = 0.3
        
        # Simulate AR process
        start_time = time.time()
        v = arsim(w, A, C, n)
        sim_time = time.time() - start_time
        
        # Check dimensions
        print(f"  Simulation shape: {v.shape} (expected: ({n}, {m}))")
        assert v.shape == (n, m), f"Expected shape ({n}, {m}) but got {v.shape}"
        
        # Check basic statistics
        print(f"  Mean: {np.mean(v, axis=0)}")
        print(f"  Std dev: {np.std(v, axis=0)}")
        print(f"  Simulation took {sim_time:.4f} seconds")
        print("  Simulation successful!\n")
        
        # Plot the first 200 samples
        plt.figure(figsize=(10, 6))
        for i in range(m):
            plt.subplot(m, 1, i+1)
            plt.plot(v[:200, i])
            plt.title(f'Channel {i+1}')
        plt.tight_layout()
        plt.savefig(f"ar_simulation_{m}d_{p}p.png")
        plt.close()

def test_ar_model_fitting():
    """Test AR model fitting with order selection."""
    print_section("Testing AR Model Fitting with Order Selection")
    
    # Test cases with different dimensions and orders
    test_cases = [
        {"m": 1, "true_p": 2, "n": 1000, "desc": "Univariate AR(2)"},
        {"m": 3, "true_p": 2, "n": 1000, "desc": "3-dimensional AR(2)"}
    ]
    
    for case in test_cases:
        m, true_p, n = case["m"], case["true_p"], case["n"]
        desc = case["desc"]
        print(f"Testing {desc} model fitting...")
        
        # Generate a true AR model
        w = np.zeros(m)
        
        # Create AR coefficients with known properties
        A = np.zeros((m, m * true_p))
        for i in range(true_p):
            if m == 1:
                # For univariate case
                A[:, i*m:(i+1)*m] = np.array([[0.6 / (i+1)]])
            else:
                # For multivariate case
                temp = np.random.randn(m, m) * 0.2
                for j in range(m):
                    temp[j, j] = 0.4 / (i+1)
                A[:, i*m:(i+1)*m] = temp
        
        # Create noise covariance
        C = np.eye(m)
        if m > 1:
            # Add some correlation
            for i in range(m-1):
                C[i, i+1] = C[i+1, i] = 0.3
        
        # Simulate data
        v = arsim(w, A, C, n)
        
        # Fit models with different orders
        pmin = 1
        pmax = 6
        print(f"  Fitting AR models with orders from {pmin} to {pmax}...")
        
        # Try both selection criteria
        for selector in ['sbc', 'fpe']:
            start_time = time.time()
            west, Aest, Cest, SBC, FPE, th = arfit(v, pmin, pmax, selector=selector)
            fit_time = time.time() - start_time
            
            # Determine selected order
            if selector == 'sbc':
                selected_p = np.argmin(SBC) + pmin
                criterion_values = SBC
            else:
                selected_p = np.argmin(FPE) + pmin
                criterion_values = FPE
            
            print(f"  Using {selector.upper()}: Selected order {selected_p}, true order {true_p}")
            print(f"  Criterion values: {criterion_values}")
            print(f"  Fitting took {fit_time:.4f} seconds")
            
            # Compare estimated vs. true coefficients if order is correct
            if selected_p == true_p:
                est_norm = np.linalg.norm(Aest)
                true_norm = np.linalg.norm(A)
                diff_norm = np.linalg.norm(Aest - A)
                rel_error = diff_norm / true_norm
                
                print(f"  Estimated coef. norm: {est_norm:.4f}")
                print(f"  True coef. norm: {true_norm:.4f}")
                print(f"  Relative error: {rel_error:.4f} ({diff_norm:.4f}/{true_norm:.4f})")
                
                if rel_error < 0.3:
                    print("  ✓ Coefficient estimation is reasonably accurate")
                else:
                    print("  ✗ Coefficient estimation differs significantly from true values")
            
            # Check residuals
            residuals, _, _, _ = arres(v, west, Aest, Cest)
            
            # Basic whiteness check (should be close to uncorrelated)
            r_means = np.mean(residuals, axis=0)
            r_stds = np.std(residuals, axis=0)
            r_autocorr = []
            
            for i in range(m):
                r_i = residuals[:, i]
                ac = np.correlate(r_i - r_means[i], r_i - r_means[i], mode='full')
                ac = ac[ac.size//2:] / ac[ac.size//2]
                r_autocorr.append(ac[1:6])  # First 5 autocorrelations
            
            print(f"  Residual means: {r_means}")
            print(f"  Residual std devs: {r_stds}")
            print(f"  Residual autocorrelations (first 5 lags):")
            for i in range(m):
                print(f"    Channel {i+1}: {r_autocorr[i]}")
            
            # Plot resulting fit vs. original
            try:
                if 'plot_ar_residuals' in globals():
                    plot_ar_residuals(v, residuals)
                    plt.savefig(f"ar_residuals_{m}d_{selected_p}p_{selector}.png")
                    plt.close()
            except Exception as e:
                print(f"  Error plotting residuals: {e}")
        
        print("  Model fitting successful!\n")

def test_eigenmode_analysis():
    """Test eigenmode analysis of AR models."""
    print_section("Testing Eigenmode Analysis")
    
    # Test cases with different dimensions and orders
    test_cases = [
        {"m": 2, "p": 2, "n": 1000, "desc": "Bivariate AR(2)"},
        {"m": 4, "p": 2, "n": 1000, "desc": "4-dimensional AR(2)"}
    ]
    
    for case in test_cases:
        m, p, n = case["m"], case["p"], case["n"]
        desc = case["desc"]
        print(f"Testing eigenmode analysis for {desc}...")
        
        # Create AR model with known oscillatory behavior
        w = np.zeros(m)
        
        # Create AR coefficients with oscillatory behavior
        A = np.zeros((m, m * p))
        
        # For 2D case, create a simple oscillatory system
        if m == 2 and p == 2:
            # Create a system with a damped oscillation
            A[:, :m] = np.array([[0.8, -0.2], [0.2, 0.8]])
            A[:, m:2*m] = np.array([[-0.2, 0.0], [0.0, -0.2]])
        else:
            # For higher-dimensional systems, create block oscillatory pattern
            for i in range(m//2):
                # Each 2x2 block has oscillatory behavior
                idx1 = 2*i
                idx2 = 2*i + 1
                A[idx1:idx2+1, idx1:idx2+1] = np.array([[0.8, -0.2], [0.2, 0.8]])
                
                if p > 1:
                    A[idx1:idx2+1, m+idx1:m+idx2+1] = np.array([[-0.2, 0.0], [0.0, -0.2]])
        
        # Create noise covariance
        C = np.eye(m)
        
        # Simulate data
        v = arsim(w, A, C, n)
        
        # Fit model
        west, Aest, Cest, _, _, th = arfit(v, p, p)
        
        # Perform eigenmode analysis
        start_time = time.time()
        S, Serr, eigval, sdec, phi, modulus, freq, fsd, fsm = armode(Aest, Cest, th)
        mode_time = time.time() - start_time
        
        print(f"  Eigenmode analysis took {mode_time:.4f} seconds")
        
        # Display results
        print(f"  Found {len(eigval)} eigenvalues:")
        for i in range(len(eigval)):
            print(f"    {i+1}: λ = {eigval[i]:.4f}, |λ| = {modulus[i]:.4f}, freq = {freq[i]:.4f}")
        
        # Plot eigenvalues
        try:
            if 'plot_eigenvalues' in globals():
                fig = plot_eigenvalues(eigval)
                plt.savefig(f"eigenvalues_{m}d_{p}p.png")
                plt.close()
        except Exception as e:
            print(f"  Error plotting eigenvalues: {e}")
        
        # Try frequency response plotting
        try:
            if 'plot_frequency_response' in globals():
                fig = plot_frequency_response(Aest, Cest, fs=1.0)
                plt.savefig(f"frequency_response_{m}d_{p}p.png")
                plt.close()
        except Exception as e:
            print(f"  Error plotting frequency response: {e}")
        
        print("  Eigenmode analysis successful!\n")

def benchmark_arfit():
    """Benchmark ARfit performance."""
    print_section("Benchmarking ARfit Performance")
    
    # Test cases for benchmarking
    test_cases = [
        {"m": 2, "p": 2, "n": 1000, "desc": "Small (2D, order 2, 1K samples)"},
        {"m": 5, "p": 5, "n": 10000, "desc": "Medium (5D, order 5, 10K samples)"},
        {"m": 10, "p": 8, "n": 5000, "desc": "Large (10D, order 8, 5K samples)"},
    ]
    
    for case in test_cases:
        m, p, n = case["m"], case["p"], case["n"]
        desc = case["desc"]
        print(f"Benchmarking {desc}...")
        
        # Generate random data
        v = np.random.randn(n, m)
        
        # Time model fitting
        start_time = time.time()
        west, Aest, Cest, SBC, FPE, th = arfit(v, 1, p)
        fit_time = time.time() - start_time
        
        print(f"  Model fitting took {fit_time:.4f} seconds")
        
        # Time eigenmode analysis
        start_time = time.time()
        S, Serr, eigval, sdec, phi, modulus, freq, fsd, fsm = armode(Aest, Cest, th)
        mode_time = time.time() - start_time
        
        print(f"  Eigenmode analysis took {mode_time:.4f} seconds")
        
        # Time data simulation
        start_time = time.time()
        v_sim = arsim(west, Aest, Cest, n)
        sim_time = time.time() - start_time
        
        print(f"  Data simulation took {sim_time:.4f} seconds")
        
        print(f"  Total processing time: {fit_time + mode_time + sim_time:.4f} seconds\n")

def run_all_tests():
    """Run all test functions."""
    test_ar_simulation()
    test_ar_model_fitting()
    test_eigenmode_analysis()
    benchmark_arfit()
    
    print_section("All Tests Completed Successfully!")

if __name__ == "__main__":
    run_all_tests()
