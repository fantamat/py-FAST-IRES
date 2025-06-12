import numpy as np
from scipy.optimize import root_scalar
from numba import njit

def Proj_Flat_Hyper_Ellips(betta_tilda, Phi_i, phi_hat, U, D, K_U, Tp_norm, num_it):
    """
    Flat hyperellipsoid projection for constraint satisfaction.
    Returns the correction E to project onto the ellipsoid boundary.
    """
    Phi = Phi_i
    D_0 = np.diag(D)
    Betta = betta_tilda
    Init_zero = 0.0

    def myfun(lambda_, Phi, D_0, Betta):
        return np.sum(Phi / (1 + lambda_ * D_0) ** 2) - Betta

    def fun(lambda_):
        return myfun(lambda_, Phi, D_0, Betta)

    # Find lambda_ini using a root-finding method (fzero equivalent)
    sol = root_scalar(fun, bracket=[0, 1e6], method='brentq')
    lambda_ini = max(sol.root, 0) if sol.converged else 0

    # Compute the correction E
    diag_term = lambda_ini / (1 + lambda_ini * D_0)
    E = (K_U.T @ np.diag(diag_term) @ U.T) @ Tp_norm
    return E
