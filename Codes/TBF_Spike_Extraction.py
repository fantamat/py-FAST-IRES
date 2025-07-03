import numpy as np
import scipy.io as sio
from numpy.linalg import svd, norm, pinv
from sklearn.decomposition import FastICA
import os
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt

def plot_eeg_signal(eeg_data, fs=1000, electrode_names=None, title="EEG Signal", show=False):
    """
    Plots EEG signals.

    Parameters:
        eeg_data (np.ndarray): EEG data of shape (electrodes, timestamps)
        fs (int): Sampling frequency in Hz (default 1000)
        electrode_names (list): List of electrode names (optional)
        title (str): Plot title
    """
    n_electrodes, n_timestamps = eeg_data.shape
    time = np.arange(n_timestamps) / fs

    plt.figure(figsize=(12, 2 * n_electrodes))
    offset = 5 * np.nanmax(np.abs(eeg_data))
    for i in range(n_electrodes):
        label = electrode_names[i] if electrode_names is not None else f"Ch {i+1}"
        plt.plot(time, eeg_data[i] + i * offset, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset")
    plt.title(title)
    plt.yticks([(i * offset) for i in range(n_electrodes)],
               electrode_names if electrode_names is not None else [f"Ch {i}" for i in range(n_electrodes)])
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()

    if show:
        plt.show()


def TBF_Spike_Extraction():
    # Load data
    data_spk_tot = sio.loadmat('Raw 273/data_spk_tot.mat')['data_spk_tot']

    data_noise = data_spk_tot.copy()

    def bandpass_filter(data, fs, lowcut, highcut, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=1)

    fs = 500  # Sampling frequency in Hz
    data_noise = bandpass_filter(data_noise, fs, 1, 50)

    plot_eeg_signal(data_noise, title="data_noise")

    Num_ICA = data_noise.shape[0]  # originally 5
    ica = FastICA(n_components=Num_ICA, max_iter=2048, whiten='unit-variance')
    activations = ica.fit_transform(data_noise.T).T  # shape: (n_components, n_samples)
    weights = ica.components_
    Topo_ICA = pinv(weights)

    test = np.abs(ica.mixing_)
    for i in range(test.shape[0]):
        test[i] = test[i] / np.linalg.norm(test[i])
    test_2 = np.max(test, axis=0)
    test_3 = np.var(test, axis=0)



    plot_eeg_signal(activations, title="activations")
    plt.show()
    
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
    Spk_Folder = 'Spikes/Spike_I'
    if not os.path.exists(Spk_Folder):
        os.makedirs(Spk_Folder)
    sio.savemat(os.path.join(Spk_Folder, 'data_spk_tot.mat'), {'data_spk_tot': data_spk_tot})
    sio.savemat(os.path.join(Spk_Folder, 'Data_Avg.mat'), {'Data_Avg': Data_Avg})
    sio.savemat(os.path.join(Spk_Folder, 'Data_Avg_clean.mat'), {'Data_Avg_clean': Data_Avg_clean})
    sio.savemat(os.path.join(Spk_Folder, 'TBF.mat'), {'TBF': TBF})
    sio.savemat(os.path.join(Spk_Folder, 'ICA.mat'), {'ICA': ICA})
    return ICA


if __name__ == "__main__":
    TBF_Spike_Extraction()