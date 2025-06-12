import numpy as np
import os
import scipy.io as sio
from numpy.linalg import svd, norm, pinv
from scipy.stats import zscore
from scipy.fft import fft, fftshift
# For ICA, use sklearn or mne
from sklearn.decomposition import FastICA

# Placeholder for topoplot and readlocs

def readlocs(filename):
    # Implement as needed for your electrode location format
    return None

def TBF_Selection_Seizure():
    # Load data
    os.chdir('../Raw 273')
    Elec_loc = readlocs('273_ele_ERD.xyz')
    data_sz_tot = sio.loadmat('data_sz_tot.mat')['data_sz_tot']
    os.chdir('../Codes')
    Samp_Rate = 500
    data_noise = data_sz_tot.copy()
    U, S, Vh = svd(data_noise, full_matrices=False)
    # Plot singular values (optional)
    # print(np.sum(np.diag(S) >= np.mean(np.diag(S))))
    Num_ICA = 25
    ica = FastICA(n_components=Num_ICA, max_iter=2048, whiten='unit-variance')
    activations = ica.fit_transform(data_noise.T).T  # shape: (n_components, n_samples)
    weights = ica.components_
    Topo_ICA = pinv(weights)
    # Variance explained (optional)
    data_recon = Topo_ICA @ weights @ data_noise
    var_explained = 100 * norm(data_recon - data_noise) / norm(data_noise)
    # Select noisy components (example indices, adjust as needed)
    ind_noise = np.array([0, 1, 4, 5, 6, 7, 8, 11, *range(12, Num_ICA)])
    Noise_comp = Topo_ICA.copy()
    Noise_comp[:, ind_noise] = 0
    data_no_blink = Noise_comp @ weights @ data_noise
    # Select non-noise components (example indices, adjust as needed)
    ind_select = np.array([2, 3, 9, 10])
    # q_factor calculation
    signal_st = Samp_Rate * 10
    signal_end = data_sz_tot.shape[1]
    q_factor = np.zeros(Num_ICA)
    for i_fac in range(Num_ICA):
        q_factor[i_fac] = np.std(np.abs(activations[i_fac, signal_st:signal_end])) / np.std(np.abs(activations[i_fac, :signal_st]))
    q_sel = q_factor[ind_select]
    # Finalize selection
    Comp_Sel = ind_select
    EEG_Sz_3_First = {
        'data': data_noise,
        'data_clean': data_no_blink,
        'Elec': Elec_loc,
        'Topo': Topo_ICA,
        'activations': activations,
        'weights': weights,
        'Comp_Sel': Comp_Sel,
        'comp_num': len(Comp_Sel)
    }
    os.chdir('../Denoised Seizures')
    sio.savemat('EEG_Sz_3_First.mat', {'EEG_Sz_3_First': EEG_Sz_3_First})
    os.chdir('../Codes')
    # Dominant Frequency Selection
    Phi = data_no_blink
    NFFT = 2 ** 10
    Phi_ab_pre = np.mean([Phi[:, i*1000:(i+1)*1000] for i in range(5)], axis=0)
    Phi_ab_post = np.mean([Phi[:, i*1000:(i+1)*1000] for i in range(5, 10)], axis=0)
    f_axis = np.linspace(-250, 250, NFFT)
    fft_pre = fftshift(np.sum(np.abs(fft(Phi_ab_pre.T, NFFT)), axis=1))
    fft_post = fftshift(np.sum(np.abs(fft(Phi_ab_post.T, NFFT)), axis=1))
    # Plotting can be done with matplotlib if needed
    # import matplotlib.pyplot as plt
    # plt.plot(f_axis, fft_pre, 'r')
    # plt.plot(f_axis, fft_post, 'g')
    # plt.xlim([0, 50])
    # plt.show()
    return EEG_Sz_3_First

if __name__ == "__main__":
    EEG_Sz_3_First = TBF_Selection_Seizure()
