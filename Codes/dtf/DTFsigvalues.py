import numpy as np
from numpy.fft import fft, ifft
from numba import njit
from typing import Optional

# You must provide a real DTF function for actual use
from Codes.DTF import DTF

def surrogate(index):
    # Randomly permute the index for phase shuffling
    return np.random.permutation(index)

def DTFsigvalues(ts, low_freq, high_freq, p, fs=400, shufftimes=1000, siglevel=0.05, handle=None):
    """
    Compute statistical significance values for relative DTF values using surrogate data.
    ts: time series (n_samples, n_channels)
    low_freq, high_freq: frequency range
    p: model order
    fs: sampling rate
    shufftimes: number of surrogates
    siglevel: significance level
    handle: progress display (ignored in Python)
    Returns: new_gamma2 (nchan, nchan, nfreq)
    """
    if fs is None:
        fs = 400
    nreps = shufftimes if shufftimes is not None else 1000
    tvalue = siglevel if siglevel is not None else 0.05

    tot_range = np.arange(low_freq, high_freq + 1)
    nfreq = len(tot_range)
    nchan = ts.shape[1]
    sig_size = int(np.floor(tvalue * nreps)) + 1
    new_gamma = np.zeros((sig_size - 1, nchan, nchan, nfreq))
    n_samples = ts.shape[0]

    for i in range(nreps):
        rate = round(100 * (i + 1) / nreps)
        print(f"Completing {rate}%")
        # Generate surrogate time series
        newts = np.zeros_like(ts)
        for j in range(nchan):
            Y = fft(ts[:, j])
            Pyy = np.abs(Y)
            Phyy = Y / (Pyy + 1e-12)  # Avoid division by zero
            index = np.arange(n_samples)
            index = surrogate(index)
            Y_new = Pyy * Phyy[index]
            newts[:, j] = np.real(ifft(Y_new))
        # Compute DTF for surrogate
        gamma2 = DTF(newts, low_freq, high_freq, p, fs)
        # Save surrogate DTF values
        new_gamma[-1, :, :, :] = gamma2
        new_gamma = np.sort(new_gamma, axis=0)[::-1]  # Descending sort
        new_gamma = new_gamma[:-1, :, :, :]  # Remove last

    # Take the surrogated DTF values at a certain significance
    new_gamma2 = np.zeros((nchan, nchan, nfreq))
    for i in range(nchan):
        for j in range(nchan):
            for k in range(nfreq):
                new_gamma2[i, j, k] = new_gamma[sig_size - 2, i, j, k]
    print("Done")
    return new_gamma2
