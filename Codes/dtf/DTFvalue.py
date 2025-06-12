import numpy as np
from numba import njit

def DTFvalue(A, low_freq, high_freq, fs):
    """
    Calculate DTF values given a coefficient matrix from a MVAR model.
    A: np.ndarray (nchan, nchan, p+1)
    low_freq: int
    high_freq: int
    fs: float
    Returns: gamma2 (nchan, nchan, nfre)
    """
    nchan = A.shape[0]
    dt = 1.0 / fs
    tot_range = np.arange(low_freq, high_freq + 1)
    nfre = len(tot_range)
    p = A.shape[2] - 1
    theta2 = np.zeros((nchan, nchan, nfre), dtype=np.complex128)
    for k in range(nfre):
        Af = np.zeros((nchan, nchan), dtype=np.complex128)
        fre = tot_range[k]
        for i in range(nchan):
            for j in range(nchan):
                for h in range(p + 1):
                    Af[i, j] -= A[i, j, h] * np.exp(-np.pi * fre * dt * (h) * 2j)
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
