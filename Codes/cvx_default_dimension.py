import numpy as np
from numba import njit

def cvx_default_dimension(sx):
    """
    Returns the first index (1-based) where sx[i] > 1, or 1 if all are 1.
    Equivalent to MATLAB's cvx_default_dimension.
    """
    sx = np.asarray(sx)
    idx = np.where(sx != 1)[0]
    if idx.size == 0:
        return 1
    else:
        return int(idx[0] + 1)  # MATLAB is 1-based

@njit
def cvx_default_dimension_numba(sx):
    for i in range(len(sx)):
        if sx[i] != 1:
            return i + 1  # 1-based index
    return 1
