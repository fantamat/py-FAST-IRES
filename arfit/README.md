# ARfit: Multivariate Autoregressive Model Fitting in Python

This package is a Python implementation of the ARfit MATLAB toolbox for multivariate autoregressive model fitting. It provides tools for:

- Estimating parameters of multivariate autoregressive (AR) models with automatic order selection
- Diagnostic checking of fitted AR models through residual analysis
- Analyzing eigenmodes and their oscillation characteristics
- Computing confidence intervals for model parameters
- Simulating multivariate AR processes
- Integration with DTF connectivity functions for the FAST-IRES project
- Visualization tools for AR model analysis

## Installation

The package is included as part of the py-FAST-IRES project. It requires:

- NumPy (>= 2.2.0)
- SciPy (>= 1.15.0)
- Matplotlib (>= 3.8.0)
- Numba (>= 0.61.0) for enhanced computational performance

## Core Functions

The ARfit package includes the following main functions:

### Model Estimation and Analysis
- `arfit`: Fits AR models with automatic order selection
- `arsim`: Simulates multivariate AR processes
- `arres`: Validates model fit through residual analysis
- `armode`: Analyzes eigenmodes, oscillation periods, and damping times
- `arconf`: Computes confidence intervals for model parameters
- `acf`: Plots autocorrelation functions

### FAST-IRES Integration
- `prepare_ar_for_dtf`: Prepares AR model parameters for DTF connectivity analysis
- `compute_dtf`: Computes Directed Transfer Function using ARfit for model estimation

### Visualization Tools
- `plot_eigenvalues`: Plots eigenvalues of the AR model in the complex plane
- `plot_frequency_response`: Visualizes the frequency response of the AR model
- `plot_ar_residuals`: Plots time series data and residuals from AR model fitting

### Numba-Accelerated Components
For improved performance, the package includes Numba-accelerated versions of:
- QR factorization for AR model fitting
- Order selection criteria computation
- Eigendecomposition of AR models
- Oscillation periods and damping times computation

## Basic Usage

Import the package:

```python
from arfit import arfit, arsim, arres, armode, arconf, acf
```

### Step 1: Simulate or load data

```python
# Simulate an AR(2) process for testing
w = np.array([0.2, 0.1])  # Intercept vector
A1 = np.array([[0.4, 0.3], [0.4, 0.5]])  # First lag coefficients
A2 = np.array([[0.2, 0.1], [0.1, 0.3]])  # Second lag coefficients
A = np.hstack((A1, A2))  # Combined coefficient matrix
C = np.array([[1.0, 0.5], [0.5, 1.2]])  # Noise covariance matrix

# Generate 1000 time steps
v = arsim(w, A, C, 1000)
```

### Step 2: Fit an AR model with automatic order selection

```python
# Try model orders from 1 to 10 and select the best
pmin = 1
pmax = 10
west, Aest, Cest, SBC, FPE, th = arfit(v, pmin, pmax)

# Determine the selected order
p_est = Aest.shape[1] // v.shape[1]
print(f"Selected model order: {p_est}")
```

### Step 3: Check model adequacy with residual analysis

```python
# Test if residuals are uncorrelated (white noise)
siglev, res = arres(west, Aest, v)
print(f"Significance level: {siglev}")
if siglev > 0.05:
    print("Residuals are uncorrelated (model is adequate)")
else:
    print("Residuals are correlated (model may be misspecified)")
    
# Plot autocorrelation function of residuals
fig, ax, rho = acf(res[:, 0, 0])
```

### Step 4: Compute confidence intervals for parameters

```python
# Get margins of error for AR coefficients and intercept
Aerr, werr = arconf(Aest, Cest, west, th)

# Display with confidence intervals
print("Intercept with 95% confidence intervals:")
for i in range(len(west)):
    print(f"  w[{i}] = {west[i, 0]:.4f} ± {werr[i, 0]:.4f}")
```

### Step 5: Analyze eigenmodes to identify oscillatory components

```python
# Compute eigenmodes, periods, and damping times
S, Serr, per, tau, exctn, lambda_vals = armode(Aest, Cest, th)

# Find oscillatory modes (non-infinite periods)
osc_idx = [i for i in range(len(exctn)) if not np.isinf(per[0, i])]
print("\nOscillatory modes:")
for i in osc_idx:
    print(f"Mode {i+1}: Period = {per[0, i]:.2f}, Damping = {tau[0, i]:.2f}, " 
          f"Excitation = {exctn[i]:.2f}")
```

## Complete Example

A full example script is provided in the `usage_example.py` file:

```python
from arfit import usage_example
usage_example.main()
```

## Integration with py-FAST-IRES

ARfit has been integrated with the DTF connectivity functions in the FAST-IRES project. This integration allows seamless use of ARfit's multivariate AR model estimation capabilities with the Directed Transfer Function (DTF) connectivity analysis.

### Using ARfit with DTF Analysis

```python
import numpy as np
from arfit import arfit, arsim
from arfit.dtf_integration import compute_dtf
import matplotlib.pyplot as plt

# Generate or load multivariate time series data
# For example, simulating a 4-channel AR(2) process with known connectivity
n_channels = 4
n_samples = 2000
order = 2

# Create AR coefficients with specific connectivity pattern:
# Channel 1 -> Channel 2, Channel 3 -> Channel 4
w = np.zeros(n_channels)
A = np.zeros((n_channels, n_channels * order))

# Create connectivity: Ch1->Ch2 and Ch3->Ch4
A[1, 0] = 0.5  # Ch1 -> Ch2 at lag 1
A[3, 2] = 0.6  # Ch3 -> Ch4 at lag 1
A[1, n_channels + 0] = 0.3  # Ch1 -> Ch2 at lag 2
A[3, n_channels + 2] = 0.2  # Ch3 -> Ch4 at lag 2

# Simulate data
data = arsim(w, A, np.eye(n_channels), n_samples)

# Compute DTF with ARfit integration
fs = 100  # Sampling frequency in Hz
low_freq = 1
high_freq = 45
gamma2 = compute_dtf(data, low_freq, high_freq, order, fs)

# Visualize DTF results
frequencies = np.linspace(low_freq, high_freq, gamma2.shape[2])
plt.figure(figsize=(12, 10))

for i in range(n_channels):
    for j in range(n_channels):
        plt.subplot(n_channels, n_channels, i*n_channels + j + 1)
        plt.plot(frequencies, gamma2[i, j, :])
        plt.ylim([0, 1])
        plt.title(f'From {j+1} to {i+1}')
        if i == n_channels-1:
            plt.xlabel('Frequency (Hz)')
        if j == 0:
            plt.ylabel('DTF')

plt.tight_layout()
plt.show()
```

### Using Visualization Tools

ARfit includes several visualization tools to help with model analysis:

```python
from arfit import arfit, arsim, arres
from arfit.visualization import plot_eigenvalues, plot_frequency_response, plot_ar_residuals

# Fit an AR model
w, A, C, SBC, FPE, th = arfit(data, 1, 5)  # Try orders 1-5

# Plot eigenvalues in complex plane
fig_eig = plot_eigenvalues(A)
fig_eig.savefig('eigenvalues.png')

# Plot frequency response
fig_freq = plot_frequency_response(A, C, fs=100)
fig_freq.savefig('frequency_response.png')

# Plot residuals and their autocorrelation
residuals, _, _, _ = arres(data, w, A, C)
fig_ts, fig_acf = plot_ar_residuals(data, residuals)
fig_ts.savefig('residuals_time_series.png')
fig_acf.savefig('residuals_autocorrelation.png')
```

## Performance Optimization with Numba

ARfit includes Numba-accelerated versions of computationally intensive functions. These optimizations are automatically used when Numba is available in your environment. To ensure optimal performance:

```python
# Make sure Numba is installed
# pip install numba

# The ARfit module will automatically use Numba accelerated functions
from arfit import arfit

# Check if Numba acceleration is available
import arfit
print(f"Using Numba acceleration: {getattr(arfit, 'HAS_NUMBA', False)}")
```

This implementation of ARfit supports the connectivity analysis needs of the FAST-IRES algorithm. It's particularly useful for:

1. Estimating MVAR models for time series of source activity
2. Performing connectivity analysis through the DTF functions
3. Identifying oscillatory components in brain signals

## Original References

This is a Python port of the original MATLAB ARfit package. If you use this code in publications, please cite the original papers:

1. Neumaier, A. and T. Schneider (2001): Estimation of parameters and eigenmodes of multivariate autoregressive models. ACM Trans. Math. Softw., 27, 27-57.

2. Schneider, T. and A. Neumaier (2001): Algorithm 808: ARfit - A Matlab package for the estimation of parameters and eigenmodes of multivariate autoregressive models. ACM Trans. Math. Softw., 27, 58-65.

## License

This software is provided under the ACM Software License Agreement, as per the original MATLAB implementation.
