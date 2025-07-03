import numpy as np
from typing import Optional, Union

def cvx_default_dimension(shape):
    """Return the first non-singleton dimension (like MATLAB)."""
    for i, s in enumerate(shape):
        if s > 1:
            return i
    return 0

def cvx_check_dimension(dim, allow_zero=False):
    """Check if the dimension is valid for numpy arrays."""
    return isinstance(dim, int) and (dim >= 0 if allow_zero else dim > 0)

def norms(x: np.ndarray):
    return np.linalg.norm(x, axis=0)
    """
    Computation of multiple vector norms (NumPy + Numba version).
    """
    if p is None:
        p = 2
    if not isinstance(p, (int, float)) or np.isnan(p):
        raise ValueError('Second argument must be a real number.')
    if p < 1:
        raise ValueError('Second argument must be between 1 and +Inf, inclusive.')

    sx = x.shape
    if dim is None:
        dim = cvx_default_dimension(sx)
    elif not cvx_check_dimension(dim, False):
        raise ValueError('Third argument must be a valid dimension.')
    elif x.size == 0 or dim >= len(sx) or sx[dim] == 1:
        p = 1

    return np.sum(np.abs(x) ** p, axis=dim) ** (1.0 / p)
