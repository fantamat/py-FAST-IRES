import numpy as np
from numba import njit
from typing import Optional

# Import the ARfit model estimation function
try:
    from arfit.dtf_integration import prepare_ar_for_dtf as arfit_implementation
    
    def arfit(ts, p):
        """
        Estimate MVAR (AR) model coefficients using ARfit.
        Args:
            ts: np.ndarray, shape (n_samples, n_channels)
            p: int, model order
        Returns:
            w: np.ndarray, noise covariance matrix (n_channels, n_channels)
            A: np.ndarray, AR coefficients (n_channels, n_channels * p)
        """
        return arfit_implementation(ts, p)
        
except ImportError:
    # Fallback to placeholder implementation if ARfit is not available
    print("Warning: ARfit not available, using fallback AR estimation method.")
    def arfit(ts, p):
        """
        Estimate MVAR (AR) model coefficients using least squares (Yule-Walker).
        Args:
            ts: np.ndarray, shape (n_samples, n_channels)
            p: int, model order
        Returns:
            w: np.ndarray, noise covariance matrix (n_channels, n_channels)
            A: np.ndarray, AR coefficients (n_channels, n_channels * p)
        """
        n_samples, n_channels = ts.shape
        # Prepare lagged data
        X = []
        Y = ts[p:]
        for i in range(1, p + 1):
            X.append(ts[p - i: n_samples - i])
        X = np.concatenate(X, axis=1)  # shape: (n_samples-p, n_channels*p)
        # Solve least squares: Y = X @ A.T + noise
        # A.T shape: (n_channels*p, n_channels)
        A_coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        A = A_coeffs.T  # shape: (n_channels, n_channels*p)
        # Estimate noise covariance
        residuals = Y - X @ A_coeffs
        w = np.cov(residuals, rowvar=False)
        return w, A

# No njit decorator on this function anymore since we need to call external functions
def DTF(ts, low_freq, high_freq, p, fs=400):
    # ts: (n_samples, n_channels)
    # Returns: gamma2 (nchan, nchan, nfre)
    tot_range = np.arange(low_freq, high_freq + 1)
    nfre = len(tot_range)
    nchan = ts.shape[1]
    dt = 1.0 / fs

    # AR model estimation (replace with actual ARfit)
    w, A = arfit(ts, p)

    # Rearrange the format of the MVAR matrix
    B = np.zeros((nchan, nchan, p + 1), dtype=np.complex128)
    B[:, :, 0] = -np.eye(nchan)
    for i in range(nchan):
        for j in range(nchan):
            # A shape: (nchan, nchan*p)
            B[i, j, 1:p+1] = A[i, j::nchan]

    theta2 = np.zeros((nchan, nchan, nfre), dtype=np.complex128)
    for k in range(nfre):
        Af = np.zeros((nchan, nchan), dtype=np.complex128)
        fre = tot_range[k]
        for i in range(nchan):
            for j in range(nchan):
                for h in range(p + 1):
                    Af[i, j] -= B[i, j, h] * np.exp(-np.pi * fre * dt * (h) * 2j)
        dett2 = np.linalg.det(Af)
        dett2 = dett2 * np.conj(dett2)
        for i in range(nchan):
            for j in range(nchan):
                Apf = np.delete(Af, i, axis=1)
                Apf = np.delete(Apf, j, axis=0)
                det2 = np.linalg.det(Apf)
                det2 = det2 * np.conj(det2)
                theta2[i, j, k] = det2 / dett2 if dett2 != 0 else 0

    gamma2 = np.zeros_like(theta2, dtype=np.complex128)
    for k in range(nfre):
        for i in range(nchan):
            denom = np.sum(theta2[i, :, k])
            for j in range(nchan):
                gamma2[i, j, k] = theta2[i, j, k] / denom if denom != 0 else 0
    return gamma2

# Example usage (replace arfit with a real implementation for actual use):
# gamma2 = DTF(ts, low_freq, high_freq, p, fs)
