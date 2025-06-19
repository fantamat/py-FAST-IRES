import numpy as np
import scipy.io as sio
import os
import time
import vtk
import matplotlib as mpl

from numpy.linalg import svd, pinv, eig
from scipy.stats import chi2
from core.norms import norms
from scipy.sparse import kron, eye
from scipy.sparse.linalg import svds
from core.FISTA_ADMM_IRES import FISTA_ADMM_IRES

from FIRES_utils import plot_mesh, plot_mesh_vtk, upsample, circshift, readlocs

# Configure visualization settings
USE_MATPLOTLIB = False
USE_VTK = True


def FIRES_Sz_Rot():
    # Set working directory
    grid_dir = 'Grid Location and Defined Parameters'
    
    K = sio.loadmat(os.path.join(grid_dir, 'UnconstCentLFD.mat'))['K']
    V = sio.loadmat(os.path.join(grid_dir, 'Gradient.mat'))['V']
    L = sio.loadmat(os.path.join(grid_dir, 'Laplacian.mat'))['L']

    # Use sparse matrix operations for better performance
    V = kron(V, eye(3))  # Expand V to 3D using sparse Kronecker product
    Norm_K = np.linalg.svd(K, compute_uv=False)[0] ** 2
    Number_edge = V.shape[0]
    Number_dipole = K.shape[1]
    
    New_Mesh = sio.loadmat(os.path.join(grid_dir, 'NewMesh.mat'))['New_Mesh']
    Location = New_Mesh[:3, :]
    
    LFD_Fxd_Vertices = sio.loadmat(os.path.join(grid_dir, 'LFD_Fxd_Vertices.mat'))
    try:
        currylfd = LFD_Fxd_Vertices['currylfd'][:,:,0]
        curryloc = LFD_Fxd_Vertices['curryloc'][:,:,0]
        currytri = LFD_Fxd_Vertices['currytri'][:,:,0]
    except:
        # Handle different array structures
        currylfd = LFD_Fxd_Vertices['currylfd']
        curryloc = LFD_Fxd_Vertices['curryloc']
        currytri = LFD_Fxd_Vertices['currytri']
    
    currytri = currytri.T - 1  # Convert to 0-indexed for Python
    
    Edge = sio.loadmat(os.path.join(grid_dir, 'Edge.mat'))['Edge']

    # Load neighbor information
    try:
        Neighbor_inclusive = sio.loadmat(os.path.join(grid_dir, 'Neighbor.mat'))['Neighbor_inclusive']
    except:
        Neighbor = sio.loadmat(os.path.join(grid_dir, 'Neighbor.mat'))

    # Color maps and perspective
    perspect = [-1, 0.25, 0.5]

    # Load electrode positions
    positions, labels = readlocs(os.path.join('Raw 273', '273_ele_ERD.xyz'))

    # Setup file paths
    Name_Sz = 'EEG_Sz_3_First'
    Var_Sz = f'{Name_Sz}.mat'
    Var_Name = f'{Name_Sz}.data_clean'
    Fig_Folder = os.path.join('Figures', Name_Sz)
    Res_Folder = os.path.join('Results', Name_Sz)
    
    # Create output folders if they don't exist
    if not os.path.exists(Fig_Folder):
        os.makedirs(Fig_Folder)
    if not os.path.exists(Res_Folder):
        os.makedirs(Res_Folder)

    # Load seizure data
    mat = sio.loadmat(os.path.join('Denoised Seizures', Var_Sz))
    Phi = mat[Var_Name]
    Phi_noisy = Phi.copy()
    Number_sensor, Number_Source = Phi_noisy.shape

    # Defining Parameters
    Activ = mat[f'{Name_Sz}.activations']
    Weights = mat[f'{Name_Sz}.weights']
    Topo_ICA = mat[f'{Name_Sz}.Topo']
    Sel_Comp = mat[f'{Name_Sz}.Comp_Sel']
    TBF = Activ[Sel_Comp, :]
    Num_TBF = TBF.shape[0]
    Noise_Topo = Topo_ICA.copy()
    Noise_Topo[:, Sel_Comp] = []
    Noise_Act = Activ.copy()
    Noise_Act[Sel_Comp, :] = []
    Noise_Space = Noise_Topo @ Noise_Act

    # Estimating Noise
    Samp_Rate = 500
    Noise_only = Phi_noisy[:, :4400]
    # Improved whitening with better numerical stability
    Sigma_inv_half = np.diag(1. / np.std(Noise_only, axis=1, ddof=1))
    Sigma_inv_half = np.diag(np.repeat(np.mean(np.diag(Sigma_inv_half)), Number_sensor))
    Phi_noisy = Phi_noisy[:, 4499:6000]  # Python 0-based
    TBF = TBF[:, 4499:6000]
    
    # Compute projections
    psi = Phi_noisy @ TBF.T @ pinv(TBF @ TBF.T)
    
    # Initialize solution
    SNR_avg = np.mean(np.std(Sigma_inv_half @ Phi_noisy, axis=1, ddof=1) ** 2) / np.mean(np.std(Sigma_inv_half @ Noise_only, axis=1, ddof=1) ** 2)
    D_W = 1.0 / (norms(K + 1e-20))
    K_n = (Sigma_inv_half @ K) * np.tile(D_W, (Number_sensor, 1))
    psi_n = Sigma_inv_half @ psi
    
    # Compute eigenvalues for regularization
    gam, _ = eig((Sigma_inv_half @ K) @ (Sigma_inv_half @ K).T)
    gam = np.diag(gam)
    Smf = 10.0
    J_ini = (K_n.T @ pinv(K_n @ K_n.T + np.mean(np.diag(gam)) * (Smf / SNR_avg) * np.eye(Number_sensor)) @ psi_n) * np.tile(D_W[:, np.newaxis], (1, Num_TBF))

    # Reweighting Parameter
    epsilon = 0.05
    prob = 0.95
    beta = chi2.ppf(prob, Number_sensor - 1)
    Number_iteration = 10
    Phi_norm = Sigma_inv_half @ Phi_noisy    
    K_norm = Sigma_inv_half @ K
    power = np.sum((Sigma_inv_half @ Noise_only) ** 2, axis=0)
    SNR = np.linalg.norm(Phi_noisy, 'fro') ** 2 / np.linalg.norm(Noise_only, 'fro') ** 2
    X, B_edges = np.histogram(power, bins=100)
    B = 0.5 * (B_edges[:-1] + B_edges[1:])  # Bin centers, like MATLAB's B output
    Sum_X = np.cumsum(X) / np.sum(X)
    ind_90 = np.where(Sum_X > 0.90)[0][0]
    ind_95 = np.where(Sum_X > 0.95)[0][0]
    CF_min = beta / (np.linalg.norm(Phi_norm, 'fro') ** 2 / Number_Source)
    CF = max(beta / B[ind_90], CF_min)
    alpha_vec = np.array([0.08])
    Num_alpha = len(alpha_vec)
    J_sol = np.zeros((Number_dipole, Num_TBF, Number_iteration, Num_alpha))
    Y_sol = np.zeros((Number_edge, Num_TBF, Number_iteration, Num_alpha))
    J_sol_2 = np.copy(J_sol)
    Y_sol_2 = np.copy(Y_sol)
    M = K.shape[1]
    N = V.shape[0]
    W = np.ones((M, Num_TBF))
    W_d = np.ones((N, Num_TBF))
    Lambda_min = Norm_K * np.sqrt(Number_sensor) / np.max(norms(K.T @ Phi_noisy))
    C_t = TBF @ TBF.T
    TBF_norm = TBF.T @ pinv(C_t) @ TBF
    T_norm = TBF.T @ pinv(C_t)
    alpha = alpha_vec
    lambda_ = 10.01 * max(1, Lambda_min)
    
    # Use sparse matrix operations for better performance
    W_v = V.T @ V
    
    # Compute largest singular value using sparse SVD for better performance with large matrices
    _, s, _ = svds(W_v, k=1)  # Get only the largest singular value
    L_v = 1.1 * s[0]  # Use the largest singular value
    
    x_0 = np.zeros((Number_dipole, Num_TBF))
    eps = 1e-3
    betta_tilda = np.linalg.norm(Phi_norm, 'fro') ** 2 / SNR / CF
    Abbas = K_norm @ K_norm.T
    U, D, _ = svd(Abbas)
    D = np.diag(D)
    K_U = U.T @ K_norm
    num_it_x = 20
    num_it_y = 20
    num_it_new = 20
    stop_crt = 1e-4
    stop_itr = 0
    weight_it = 0
    max_weight_itr_num = Number_iteration
    t1 = time.time()
    
    for i_alpha in range(Num_alpha):
        alpha = alpha_vec[i_alpha]
        x_0 = J_ini
        stop_itr = 0
        weight_it = 0
        W = np.ones((M, Num_TBF))
        W_d = np.ones((N, Num_TBF))
        
        while not stop_itr:
            weight_it += 1
            if weight_it > max_weight_itr_num:
                stop_itr = 1
                
            # Call FISTA_ADMM_IRES function for optimization
            J, Y = FISTA_ADMM_IRES(Phi_norm, TBF, TBF_norm, T_norm, alpha, lambda_, W_v, V, L_v, x_0, W, W_d, eps, betta_tilda, U, D, K_norm, K_U, num_it_x, num_it_y, num_it_new)
            x_0 = J
            J_sol[:, :, weight_it - 1, i_alpha] = J
            Y_sol[:, :, weight_it - 1, i_alpha] = Y
            
            # Reshape for proper normalization
            J_n = J.reshape((3, Number_dipole // 3, Num_TBF), order='F')
            Ab_J = np.squeeze(norms(J_n)) + 1e-20
            Y_n = (V @ J).reshape((3, Number_edge // 3, Num_TBF), order='F')
            Ab_Y = np.squeeze(norms(Y_n)) + 1e-20
            
            # Weight updating
            W_old = W.copy()
            W_tr = 1.0 / (Ab_J / np.tile(np.max(Ab_J, axis=0), (Number_dipole // 3, 1)) + (epsilon + 1e-16))
            
            upsample_result = upsample(W_tr, 3)
            W = circshift(upsample_result, [0, 0]) + circshift(upsample_result, [1, 0]) + circshift(upsample_result, [2, 0])
            
            W_d_old = W_d.copy()
            W_d_tr = 1.0 / (Ab_Y / np.tile(np.max(Ab_Y, axis=0), (Number_edge // 3, 1)) + (epsilon + 1e-16))
            
            upsample_result_d = upsample(W_d_tr, 3)
            W_d = circshift(upsample_result_d, [0, 0]) + circshift(upsample_result_d, [1, 0]) + circshift(upsample_result_d, [2, 0])
            
            # Check convergence
            if np.linalg.norm(W - W_old) / np.linalg.norm(W_old) < stop_crt and np.linalg.norm(W_d - W_d_old) / np.linalg.norm(W_d_old) < stop_crt:
                stop_itr = 1
            
            # Generate visualizations
            print(f"Generating visualizations for iteration {weight_it}...")
            
            # Choose visualization method based on configuration
            if USE_MATPLOTLIB:
                plot_mesh(curryloc, currytri, Ab_J, J, weight_it, Fig_Folder, perspect)
            if USE_VTK:
                plot_mesh_vtk(curryloc, currytri, Ab_J, J, weight_it, Fig_Folder, perspect)
    
    # Calculate and output elapsed time
    t_elaps = time.time() - t1
    print(f"Elapsed time (hours): {t_elaps / 3600}")
    
    # Save results
    sio.savemat(os.path.join(Res_Folder, 'J_sol_1st.mat'), {'J_sol': J_sol})
    sio.savemat(os.path.join(Res_Folder, 'TBF_1st.mat'), {'TBF': TBF})
    sio.savemat(os.path.join(Res_Folder, 'Phi_norm_1st.mat'), {'Phi_norm': Phi_norm})
    sio.savemat(os.path.join(Res_Folder, 'Phi_noisy.mat'), {'Phi_noisy': Phi_noisy})
    sio.savemat(os.path.join(Res_Folder, 'Sigma_inv_half.mat'), {'Sigma_inv_half': Sigma_inv_half})
    
    # Save parameters
    Parameters = {
        'alpha': alpha,
        'betta': betta_tilda,
        'num_it_weight': max_weight_itr_num,
        'alpha_vec': alpha_vec,
        'lambda': lambda_,
        'num_it_x': num_it_x,
        'num_it_y': num_it_y,
        'eps': eps,
        'stop_crt_wt': stop_crt,
        'epsilon': epsilon,
        'time': t_elaps,
        'iteration': weight_it
    }
    sio.savemat(os.path.join(Res_Folder, 'Parameters.mat'), {'Parameters': Parameters})


if __name__ == "__main__":
    FIRES_Sz_Rot()