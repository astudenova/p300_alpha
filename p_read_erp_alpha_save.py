import os

import numpy as np
from scipy.signal import butter, filtfilt
from tools_general import load_json, save_pickle, list_from_many
from tools_signal import compute_envelope, from_epoch_to_cont, filter_in_low_frequency
from tools_lifedataset import read_erp
from scipy.stats import pearsonr

import matplotlib as mpl

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

dir_save = load_json('dirs_files', os.getcwd())['dir_save']

ids = load_json('ids', os.getcwd())

for i_subj, subj in enumerate(ids):
    # read ERs
    erp_s, erp_t, _ = read_erp(subj, decim=1, notch=True)
    # low-pass the ER
    fs = erp_t.info['sfreq']
    erp_t_filt = filter_in_low_frequency(erp_t.get_data(picks='eeg'), fs, padlen=None)
    erp_s_filt = filter_in_low_frequency(erp_s.get_data(picks='eeg'), fs, padlen=None)


    # compute single-trial alpha amplitude envelope
    env_s = compute_envelope(erp_s.get_data(picks='eeg'), n_epoch=erp_s.events.shape[0],
                             n_ch=len(erp_s.ch_names) - 3, fs=fs, subtract=True)
    print('Envelope for standard stimulus is computed.')

    env_t = compute_envelope(erp_t.get_data(picks='eeg'), n_epoch=erp_t.events.shape[0],
                             n_ch=len(erp_t.ch_names) - 3, fs=fs, subtract=True)
    print('Envelope for target stimulus is computed.')

    # create concatenated signal for target ER
    n_ch = env_t.shape[1]
    n_epoch = env_t.shape[0]
    n_times = erp_t_filt.shape[2] - 200  # cut 100 ms from each end to avoid edge artifacts
    erp_t_filt_flat = from_epoch_to_cont(erp_t_filt, n_ch, n_epoch, wings=100)
    env_t_flat = from_epoch_to_cont(env_t, n_ch, n_epoch, wings=100)

    # compute correlation for target ER
    corr_t = [pearsonr(erp_t_filt_flat[i], env_t_flat[i])[0] for i in range(n_ch)]

    # create concatenated signal for standard ER with picking the epochs
    s_sample = np.random.random_integers(low=0, high=erp_s_filt.shape[0] - 1, size=(n_epoch,))
    erp_s_filt_flat = from_epoch_to_cont(erp_s_filt[s_sample], n_ch, n_epoch, wings=100)
    env_s_flat = from_epoch_to_cont(env_s[s_sample], n_ch, n_epoch, wings=100)

    # compute correlation for standard ER
    corr_s = [pearsonr(erp_s_filt_flat[i], env_s_flat[i])[0] for i in range(n_ch)]

    save_pickle(subj + '_erp_t', dir_save, np.mean(erp_t_filt, axis=0))
    save_pickle(subj + '_erp_s', dir_save, np.mean(erp_s_filt, axis=0))
    save_pickle(subj + '_env_t', dir_save, np.mean(env_t, axis=0))
    save_pickle(subj + '_env_s', dir_save, np.mean(env_s, axis=0))
    save_pickle(subj + '_corr_t', dir_save, corr_t)
    save_pickle(subj + '_corr_s', dir_save, corr_s)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# collect all the data into single arrays
erp_t_all, _ = list_from_many(ids, dir_save, '_erp_t', 'pickle')
erp_s_all, _ = list_from_many(ids, dir_save, '_erp_s', 'pickle')
env_t_all, _ = list_from_many(ids, dir_save, '_env_t', 'pickle')
env_s_all, _ = list_from_many(ids, dir_save, '_env_s', 'pickle')
corr_t_all, _ = list_from_many(ids, dir_save, '_corr_t', 'pickle')
corr_s_all, _ = list_from_many(ids, dir_save, '_corr_s', 'pickle')
