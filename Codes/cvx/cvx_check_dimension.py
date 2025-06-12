import numpy as np
from numba import njit

def cvx_check_dimension(x, zero_ok=False):
    """
    Verifies that the input is a valid dimension (positive integer scalar).
    If zero_ok is True, zero is also accepted.
    """
    if isinstance(x, (int, np.integer, float, np.floating)) and np.isreal(x) and x < np.inf and x == int(x):
        return (x > 0) or zero_ok
    else:
        return False

# Numba-accelerated version for scalar integer input
@njit
def cvx_check_dimension_numba(x, zero_ok=False):
    if x < float('inf') and x == int(x):
        return (x > 0) or zero_ok
    else:
        return False
