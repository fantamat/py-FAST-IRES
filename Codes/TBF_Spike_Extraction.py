import numpy as np
import scipy.io as sio
from numpy.linalg import svd, norm, pinv
from sklearn.decomposition import FastICA
import os

# Placeholder for readlocs and topoplot

def readlocs(filename):
    # Implement as needed for your electrode location format
    return None

def TBF_Spike_Extraction():
    # Load data
    os.chdir('../Raw 273')
    Elec_loc = readlocs('273_ele_ERD.xyz')
    data_spk_tot = sio.loadmat('data_spk_tot.mat')['data_spk_tot']
    os.chdir('../Codes')
    Samp_Rate = 500
    data_noise = data_spk_tot.copy()
    U, S, Vh = svd(data_noise, full_matrices=False)
    Num_ICA = 15
    ica = FastICA(n_components=Num_ICA, max_iter=2048, whiten='unit-variance')
    activations = ica.fit_transform(data_noise.T).T  # shape: (n_components, n_samples)
    weights = ica.components_
    Topo_ICA = pinv(weights)
    # Variance explained (optional)
    data_recon = Topo_ICA @ weights @ data_noise
    var_explained = 100 * norm(data_recon - data_noise) / norm(data_noise)
    # Select noisy components (example: last 7 components)
    Noise_comp = Topo_ICA.copy()
    Noise_comp[:, 8:] = 0
    data_spk_clean = Noise_comp @ weights @ data_noise
    # Average the epochs
    Win_Len = 2001
    Num_Spk = data_spk_tot.shape[1] // Win_Len
    Act = np.zeros((Num_ICA, Win_Len))
    for i_epoch in range(Num_Spk):
        Act += activations[:, i_epoch * Win_Len : (i_epoch + 1) * Win_Len]
    Act = Act / Num_Spk
    # q_factor - See which components are active around spike peak point
    ind_select = np.arange(8)
    q_factor = np.zeros(Num_ICA)
    signal_st = 800
    signal_end = 1250
    for i_fac in range(Num_ICA):
        q_factor[i_fac] = np.std(Act[i_fac, signal_st:signal_end]) / np.std(np.concatenate([Act[i_fac, :signal_st], Act[i_fac, signal_end:]]))
    q_sel = q_factor[ind_select]
    q_gen = q_factor[8:]
    # Select relevant Components after inspection - Finalize selection and save
    TBF_sel = [0, 1, 2, 4, 5]  # Python 0-based
    TBF = Act[TBF_sel, :]
    # Plot Average/De-noised Spikes
    Data_Avg = np.zeros((data_spk_tot.shape[0], Win_Len))
    for i_epoch in range(Num_Spk):
        Data_Avg += data_spk_clean[:, i_epoch * Win_Len : (i_epoch + 1) * Win_Len]
    Data_Avg = Data_Avg / Num_Spk
    Data_Avg_proj = Data_Avg @ TBF.T @ pinv(TBF @ TBF.T) @ TBF
    N_wht = Data_Avg - Data_Avg_proj
    Data_Avg_clean = Data_Avg_proj
    ICA = {
        'Number': Num_ICA,
        'act': activations,
        'Act': Act,
        'Topo': Topo_ICA,
        'q_sel': q_sel,
        'q_gen': q_gen,
        'Noise_comp': np.arange(8, Num_ICA),
        'TBF_comp': TBF_sel,
        'Win_Len': Win_Len
    }
    os.chdir('../Spikes')
    Spk_Folder = '../Spikes/Spike_I'
    if not os.path.exists(Spk_Folder):
        os.makedirs(Spk_Folder)
    os.chdir(Spk_Folder)
    sio.savemat('data_spk_tot.mat', {'data_spk_tot': data_spk_tot})
    sio.savemat('Data_Avg.mat', {'Data_Avg': Data_Avg})
    sio.savemat('Data_Avg_clean.mat', {'Data_Avg_clean': Data_Avg_clean})
    sio.savemat('TBF.mat', {'TBF': TBF})
    sio.savemat('ICA.mat', {'ICA': ICA})
    os.chdir('../../Codes')
    return ICA
