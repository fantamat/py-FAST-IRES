"""
Comparison test between ARfit Python implementation and MATLAB original.

This script is designed to validate the Python implementation of ARfit by 
comparing its results with the output from the original MATLAB version. The test
uses pre-generated data and expected results from MATLAB to ensure compatibility.
"""

import numpy as np
import scipy.io as sio
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure the package is in the path
sys.path.append(os.path.abspath('.'))

# Import ARfit modules
try:
    from arfit import arfit, arsim, arres, armode, arconf
    from arfit.visualization import plot_eigenvalues, plot_frequency_response, plot_ar_residuals
    print("Successfully imported all ARfit modules!")
except ImportError as e:
    print(f"Error importing ARfit modules: {e}")
    sys.exit(1)

def load_matlab_data(mat_file):
    """
    Load test data and expected results from MATLAB.
    
    If the MAT file doesn't exist, generate synthetic data and note that
    MATLAB comparison will not be possible.
    """
    if os.path.exists(mat_file):
        print(f"Loading MATLAB data from {mat_file}")
        matlab_data = sio.loadmat(mat_file)
        return matlab_data
    else:
        print(f"MATLAB data file {mat_file} not found.")
        print("Generating synthetic data for testing the Python implementation only.")
        # Generate synthetic data with known properties
        m = 3  # Number of channels
        p = 2  # Model order
        n = 1000  # Number of time points
        
        # Create stable AR coefficients
        A1 = np.array([
            [0.5, 0.1, 0],
            [0.2, 0.5, 0.1],
            [0, 0.1, 0.5]
        ])
        A2 = np.array([
            [0.1, 0.05, 0],
            [0.05, 0.1, 0],
            [0, 0.05, 0.1]
        ])
        
        A = np.hstack((A1, A2))
        
        # Create intercept vector
        w = np.array([0.1, 0.2, 0.3])
        
        # Create covariance matrix
        C = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.3],
            [0.1, 0.3, 1.0]
        ])
        
        # Simulate AR process
        v = arsim(w, A, C, n)
        
        # Return as a dictionary similar to what we'd get from MATLAB
        synthetic_data = {
            'v': v,
            'A_true': A,
            'w_true': w,
            'C_true': C,
            'matlab_avail': False  # Flag indicating this is not MATLAB data
        }
        
        return synthetic_data

def compare_ar_fitting(data):
    """
    Compare AR model fitting between Python and MATLAB implementations.
    """
    print("\n" + "="*80)
    print("Comparing AR Model Fitting")
    print("="*80)
    
    # Get data
    v = data['v']
    
    # Check if we have MATLAB results for comparison
    matlab_avail = data.get('matlab_avail', False)
    
    if matlab_avail:
        # Get expected MATLAB results
        A_matlab = data['A_matlab']
        w_matlab = data['w_matlab']
        C_matlab = data['C_matlab']
        sbc_matlab = data['SBC_matlab'].flatten()
        fpe_matlab = data['FPE_matlab'].flatten()
        
        print(f"MATLAB Coef. Matrix Shape: {A_matlab.shape}")
        print(f"MATLAB Intercept Shape: {w_matlab.shape}")
        print(f"MATLAB SBC: {sbc_matlab}")
        print(f"MATLAB FPE: {fpe_matlab}")
    
    # Run Python ARfit
    pmin = 1
    pmax = 5
    
    print(f"\nFitting AR model with orders {pmin} to {pmax}...")
    
    for selector in ['sbc', 'fpe']:
        print(f"\nUsing {selector.upper()} criterion:")
        
        # Fit model
        w_py, A_py, C_py, SBC_py, FPE_py, th_py = arfit(v, pmin, pmax, selector=selector)
        
        # Get selected order
        if selector == 'sbc':
            best_idx = np.argmin(SBC_py)
            criterion = SBC_py
        else:
            best_idx = np.argmin(FPE_py)
            criterion = FPE_py
        
        selected_p = best_idx + pmin
        
        print(f"  Selected order: {selected_p}")
        print(f"  Criterion values: {criterion}")
        print(f"  Coef. Matrix Shape: {A_py.shape}")
        print(f"  Intercept Shape: {w_py.shape}")
        
        # Compare with MATLAB if available
        if matlab_avail:
            # We need to account for possible sign flips and different order selection
            if selector == 'sbc':
                matlab_best_idx = np.argmin(sbc_matlab)
            else:
                matlab_best_idx = np.argmin(fpe_matlab)
            
            matlab_selected_p = matlab_best_idx + pmin
            print(f"  MATLAB selected order: {matlab_selected_p}")
            
            if selected_p == matlab_selected_p:
                # Compare coefficients and intercept
                # We use norms rather than element-wise comparison due to possible sign differences
                A_py_norm = np.linalg.norm(A_py)
                A_matlab_norm = np.linalg.norm(A_matlab)
                A_diff_norm = np.linalg.norm(A_py - A_matlab)
                A_rel_diff = A_diff_norm / max(A_py_norm, A_matlab_norm)
                
                w_py_norm = np.linalg.norm(w_py)
                w_matlab_norm = np.linalg.norm(w_matlab)
                w_diff_norm = np.linalg.norm(w_py - w_matlab)
                w_rel_diff = w_diff_norm / max(w_py_norm, w_matlab_norm)
                
                C_py_norm = np.linalg.norm(C_py)
                C_matlab_norm = np.linalg.norm(C_matlab)
                C_diff_norm = np.linalg.norm(C_py - C_matlab)
                C_rel_diff = C_diff_norm / max(C_py_norm, C_matlab_norm)
                
                print(f"  Coefficient matrix relative difference: {A_rel_diff:.4f}")
                print(f"  Intercept relative difference: {w_rel_diff:.4f}")
                print(f"  Covariance matrix relative difference: {C_rel_diff:.4f}")
                
                if A_rel_diff < 0.1 and w_rel_diff < 0.1 and C_rel_diff < 0.1:
                    print("  ✓ Results match within tolerance!")
                else:
                    print("  ✗ Results differ more than expected.")
            else:
                print(f"  Different order selected: Python={selected_p}, MATLAB={matlab_selected_p}")
        
        # Test residuals
        res, siglev, _, _ = arres(v, w_py, A_py, C_py)
        print(f"  Residual test significance level: {siglev:.4f}")
        print(f"  Uncorrelated residuals: {'Yes' if siglev > 0.05 else 'No'}")
        
        # Show results
        print(f"  First few coefficients:\n{A_py[:, :min(6, A_py.shape[1])]}")
        print(f"  Intercept:\n{w_py}")

def compare_eigenmodes(data):
    """
    Compare eigenmode analysis between Python and MATLAB implementations.
    """
    print("\n" + "="*80)
    print("Comparing Eigenmode Analysis")
    print("="*80)
    
    # Get data and fit model
    v = data['v']
    
    # Check if we have MATLAB results
    matlab_avail = data.get('matlab_avail', False)
    
    # Fit AR model
    order = 2  # Use fixed order for comparison
    w_py, A_py, C_py, _, _, th_py = arfit(v, order, order)
    
    # Analyze eigenmodes
    print(f"\nAnalyzing eigenmodes for order {order} AR model...")
    S, Serr, eigval, sdec, phi, modulus, freq, fsd, fsm = armode(A_py, C_py, th_py)
    
    # Display results
    print(f"Found {len(eigval)} eigenvalues:")
    for i in range(len(eigval)):
        print(f"  {i+1}: λ = {eigval[i]:.4f}, |λ| = {modulus[i]:.4f}, freq = {freq[i]:.4f}")
    
    # Compare with MATLAB if available
    if matlab_avail and 'eigval_matlab' in data:
        eigval_matlab = data['eigval_matlab'].flatten()
        modulus_matlab = data['modulus_matlab'].flatten()
        freq_matlab = data['freq_matlab'].flatten()
        
        print("\nComparing eigenvalues with MATLAB:")
        
        # Sort both sets of eigenvalues by magnitude for comparison
        idx_py = np.argsort(modulus)[::-1]
        idx_matlab = np.argsort(modulus_matlab)[::-1]
        
        eigval_py_sorted = eigval[idx_py]
        eigval_matlab_sorted = eigval_matlab[idx_matlab]
        
        modulus_py_sorted = modulus[idx_py]
        modulus_matlab_sorted = modulus_matlab[idx_matlab]
        
        freq_py_sorted = freq[idx_py]
        freq_matlab_sorted = freq_matlab[idx_matlab]
        
        # Compare
        match_count = 0
        for i in range(min(len(eigval_py_sorted), len(eigval_matlab_sorted))):
            mod_diff = abs(modulus_py_sorted[i] - modulus_matlab_sorted[i]) / max(modulus_py_sorted[i], modulus_matlab_sorted[i])
            freq_diff = abs(freq_py_sorted[i] - freq_matlab_sorted[i]) if freq_py_sorted[i] > 0 and freq_matlab_sorted[i] > 0 else 0
            
            if mod_diff < 0.1 and freq_diff < 0.1:
                match_count += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} Python: |λ| = {modulus_py_sorted[i]:.4f}, freq = {freq_py_sorted[i]:.4f} | "
                  f"MATLAB: |λ| = {modulus_matlab_sorted[i]:.4f}, freq = {freq_matlab_sorted[i]:.4f}")
        
        match_pct = match_count / min(len(eigval_py_sorted), len(eigval_matlab_sorted)) * 100
        print(f"\n  {match_count} out of {min(len(eigval_py_sorted), len(eigval_matlab_sorted))} modes match ({match_pct:.1f}%)")
    
    # Plot eigenvalues
    try:
        fig = plot_eigenvalues(eigval)
        fig.savefig('eigenvalues_comparison.png')
        print("\nSaved eigenvalues plot to eigenvalues_comparison.png")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting eigenvalues: {e}")
    
    return eigval, modulus, freq

def compare_simulation(data):
    """
    Compare AR process simulation between Python and MATLAB implementations.
    """
    print("\n" + "="*80)
    print("Comparing AR Process Simulation")
    print("="*80)
    
    # Define AR parameters for simulation
    m = 3  # Number of channels
    p = 2  # Model order
    n = 1000  # Number of time points
    
    # Create stable AR coefficients
    A1 = np.array([
        [0.5, 0.1, 0],
        [0.2, 0.5, 0.1],
        [0, 0.1, 0.5]
    ])
    A2 = np.array([
        [0.1, 0.05, 0],
        [0.05, 0.1, 0],
        [0, 0.05, 0.1]
    ])
    
    A = np.hstack((A1, A2))
    
    # Create intercept vector
    w = np.array([0.1, 0.2, 0.3])
    
    # Create covariance matrix
    C = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.3],
        [0.1, 0.3, 1.0]
    ])
    
    print(f"Simulating {m}-dimensional AR({p}) process with {n} time points...")
    
    # Simulate AR process
    v_sim = arsim(w, A, C, n)
    
    # Check basic properties
    print(f"Simulated data shape: {v_sim.shape}")
    print(f"Mean values: {np.mean(v_sim, axis=0)}")
    print(f"Std deviation: {np.std(v_sim, axis=0)}")
    
    # Compare with expected mean (should be close to w for a stationary process)
    mean_diff = np.linalg.norm(np.mean(v_sim, axis=0) - w) / np.linalg.norm(w)
    print(f"Relative difference to expected mean: {mean_diff:.4f}")
    
    # Check if time series has expected statistical properties
    # Fit AR model to the simulated data
    w_est, A_est, C_est, _, _, _ = arfit(v_sim, p, p)
    
    # Compare estimated coefficients with true ones
    A_diff = np.linalg.norm(A_est - A) / np.linalg.norm(A)
    w_diff = np.linalg.norm(w_est - w) / np.linalg.norm(w)
    C_diff = np.linalg.norm(C_est - C) / np.linalg.norm(C)
    
    print(f"Relative differences between true and estimated parameters:")
    print(f"  AR coefficients: {A_diff:.4f}")
    print(f"  Intercept: {w_diff:.4f}")
    print(f"  Covariance: {C_diff:.4f}")
    
    if A_diff < 0.2 and w_diff < 0.2 and C_diff < 0.2:
        print("✓ Simulation produces expected statistical properties!")
    else:
        print("✗ Simulation results differ from expected values more than anticipated.")
    
    # Plot the simulated data
    try:
        plt.figure(figsize=(10, 6))
        for i in range(m):
            plt.subplot(m, 1, i+1)
            plt.plot(v_sim[:200, i])
            plt.title(f'Channel {i+1}')
            plt.xlabel('Time' if i == m-1 else '')
            plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig('simulation_comparison.png')
        print("\nSaved simulation plot to simulation_comparison.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting simulation: {e}")
    
    return v_sim

def run_comprehensive_tests():
    """
    Run all tests to validate the ARfit Python implementation.
    """
    print("\n" + "="*80)
    print(" ARfit Python Implementation Validation Test")
    print("="*80 + "\n")
    
    # Try to load MATLAB test data
    mat_file = 'matlab_arfit_test_data.mat'
    
    try:
        # Check if we're in the right directory
        if not Path('arfit').is_dir():
            print("Warning: Script may not be running from the correct directory.")
            print(f"Current directory: {os.getcwd()}")
        
        data = load_matlab_data(mat_file)
        
        # Run comparison tests
        compare_ar_fitting(data)
        eigvals, modulus, freqs = compare_eigenmodes(data)
        sim_data = compare_simulation(data)
        
        # Test visualization functions
        print("\n" + "="*80)
        print("Testing Visualization Functions")
        print("="*80)
        
        # Get a model for visualization tests
        v = data['v']
        w_py, A_py, C_py, _, _, th_py = arfit(v, 2, 2)
        
        try:
            # Plot eigenvalues
            fig = plot_eigenvalues(eigvals)
            fig.savefig('eigenvalues_viz_test.png')
            print("✓ Eigenvalues plot created successfully")
            plt.close(fig)
            
            # Plot frequency response
            fig = plot_frequency_response(A_py, C_py, fs=100)
            fig.savefig('frequency_response_test.png')
            print("✓ Frequency response plot created successfully")
            plt.close(fig)
            
            # Plot residuals
            res, _, _, _ = arres(v, w_py, A_py, C_py)
            fig_ts, fig_acf = plot_ar_residuals(v, res)
            fig_ts.savefig('residuals_time_series_test.png')
            fig_acf.savefig('residuals_acf_test.png')
            print("✓ Residual plots created successfully")
            plt.close(fig_ts)
            plt.close(fig_acf)
        
        except Exception as e:
            print(f"Error in visualization tests: {e}")
        
        print("\n" + "="*80)
        print(" All Tests Completed ")
        print("="*80 + "\n")
    
    except Exception as e:
        print(f"Error running tests: {e}")

if __name__ == "__main__":
    run_comprehensive_tests()
