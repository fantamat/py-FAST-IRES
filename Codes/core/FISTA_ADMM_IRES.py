import numpy as np
from core.norms import norms

from core.Proj_Flat_Hyper_Ellips import Proj_Flat_Hyper_Ellips

# Placeholders for MATLAB-specific functions
# You must implement or replace these with your own Python equivalents
# e.g., wthresh, Proj_Flat_Hyper_Ellips

def wthresh(x, mode, thresh):
    """
    Thresholding function similar to MATLAB's wthresh.
    mode: 's' for soft, 'h' for hard thresholding.
    """
    if mode == 's':  # Soft threshold
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
    elif mode == 'h':  # Hard threshold
        return x * (np.abs(x) > thresh)
    else:
        raise ValueError("Second argument must be 's' (soft) or 'h' (hard)")


def FISTA_ADMM_IRES(Phi, TBF, TBF_norm, T_norm, alpha, lambda_, W_v, V, L_v, x_0, W, W_d, epsilon, betta_tilda, U, D, K, K_U, num_it_x, num_it_y, num_it_new):
    Alpha = np.diag(np.sum(np.abs(x_0), axis=0) / np.max(np.sum(np.abs(x_0), axis=0)))
    y_cond = True
    y_it = 0
    x_tild = x_0.copy()
    x_old = x_0.copy()
    y = V @ x_tild
    u = np.zeros_like(y)
    e_rel = 1e-4
    e_abs = 1e-8
    tau = 2
    mu = 10

    while y_cond:
        y_it += 1
        print(f"y_it: {y_it}")
        if y_it > num_it_y:
            y_cond = False
        t_old = 1
        x_it = 0
        x_cond = True
        while x_cond:
            x_it += 1
            if x_it > num_it_x:
                x_cond = False
            thresh = (W @ Alpha) * (alpha / lambda_ / L_v)
            x_new_tran = wthresh(x_tild - (W_v @ x_tild - V.T @ (y + u)) / L_v, 's', thresh)
            phi_hat = Phi - K @ x_new_tran @ TBF
            Phi_i = norms((U.T @ phi_hat @ TBF_norm).T) ** 2
            Tp_norm = phi_hat @ T_norm
            N_Phi = phi_hat - phi_hat @ TBF_norm
            N_F = np.linalg.norm(N_Phi, 'fro') ** 2
            if np.linalg.norm(phi_hat, 'fro') ** 2 <= betta_tilda:
                x_new_er = np.zeros_like(x_new_tran)
            else:
                x_new_er = Proj_Flat_Hyper_Ellips(betta_tilda - N_F, Phi_i, phi_hat, U, D, K_U, Tp_norm, num_it_new)
            x_new = x_new_tran + x_new_er
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            x_tild = x_new + ((t_old - 1) / t_new) * (x_new - x_old)
            if np.linalg.norm(x_new - x_old, 'fro') / np.linalg.norm(x_new, 'fro') < epsilon:
                x_cond = False
            x_old = x_new
            t_old = t_new
        y_new = wthresh(V @ x_new - u, 's', W_d / lambda_)
        y_new_norm = np.linalg.norm(y_new, 'fro')
        print(f"y_new_norm: {y_new_norm}")
        if y_new_norm > 0 and np.linalg.norm(y - y_new, 'fro') / y_new_norm < epsilon:
            y_cond = False
        prim_res = np.linalg.norm(y_new - V @ x_new, 'fro')
        dual_res = np.linalg.norm(lambda_ * (V.T) @ (y - y_new), 'fro')
        e_prim = np.sqrt(y_new.size) * e_abs + e_rel * max(np.sum(norms(V @ x_new)), np.sum(norms(y_new)))
        e_dual = np.sqrt(x_new.size) * e_abs + e_rel * np.sum(norms(V.T @ y_new))
        if prim_res <= e_prim and dual_res <= e_dual:
            y_cond = False
        y = y_new
        u = u + y - V @ x_new
        # Adaptive lambda update (commented out in original)
        # if prim_res > mu * dual_res:
        #     lambda_ = lambda_ * tau
        #     u = u / tau
        # elif dual_res > mu * prim_res:
        #     lambda_ = lambda_ / tau
        #     u = u * tau
    x = wthresh(x_new, 's', (W @ Alpha) * (alpha / lambda_ / L_v))
    y_new = V @ x_new - u
    y = wthresh(y_new, 's', W_d / lambda_)
    return x, y
