"""
This pipeline computes baseline-shift indices for sensor space.
"""
import os
import numpy as np
from scipy.signal import welch
from tools_general import load_json, save_pickle, list_from_many
from tools_lifedataset import read_rest
from tools_signal import peak_in_spectrum, bsi

dir_save = load_json('settings/dirs_files', os.getcwd())['dir_save']
dir_data = load_json('settings/dirs_files', os.getcwd())['dir_data']
ids = load_json('settings/ids', os.getcwd())
markers_rest = load_json('settings/markers_rest', os.getcwd())

for i_subj, subj in enumerate(ids):

    raw = read_rest(subj, dir_data, markers_rest[subj])

    fs = raw.info['sfreq']
    n_ch = len(raw.ch_names) - 3
    raw_data = raw.get_data(picks='eeg')
    n_times = raw_data.shape[1]
    bsi_sen = np.zeros((n_ch,))

    # compute spectrum
    f, Pxx = welch(raw_data, fs=fs, nperseg=10 * fs, noverlap=5 * fs, nfft=10 * fs, detrend=False)

    # compute BSI for each channel
    for i in range(n_ch):
        X = raw_data[i]
        alpha_peak = peak_in_spectrum(Pxx[i], f)
        bsi_sen[i] = bsi(X, fs, alpha_peak=alpha_peak)
        print('BSI is computed for sensor ' + raw.ch_names[i])

    save_pickle(subj + '_bsi', dir_save, bsi_sen)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# gather all BSI into a single file
bsi_all, _ = list_from_many(ids, dir_save, '_bsi', 'pickle')
save_pickle('bsi_all', dir_save, bsi_all)
