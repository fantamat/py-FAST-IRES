import numpy as np
from numba import njit

@njit
def DTFsigtest(gamma2, gamma2_sig):
    """
    Perform significance test for the calculated DTF values.
    gamma2: np.ndarray (nchan, nchan, nfreq)
    gamma2_sig: np.ndarray (nchan, nchan, nfreq)
    Returns: gamma2 with insignificant values set to 0.
    """
    nchan = gamma2.shape[0]
    nfreq = gamma2.shape[2]
    for i in range(nchan):
        for j in range(nchan):
            for k in range(nfreq):
                if i != j:
                    if gamma2[i, j, k] < gamma2_sig[i, j, k]:
                        gamma2[i, j, k] = 0
                else:
                    gamma2[i, j, k] = 0
    return gamma2
