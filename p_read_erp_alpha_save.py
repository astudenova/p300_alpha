"""
This pipeline reads the data files, computes ER and alpha amplitude envelope for target and standard stimuli.
"""
import os
import numpy as np
from scipy.stats import pearsonr
from tools_general import load_json, save_pickle, list_from_many
from tools_signal import compute_envelope, from_epoch_to_cont, filter_in_low_frequency
from tools_lifedataset import read_erp

dir_save = load_json('settings/dirs_files.json', os.getcwd())['dir_save']
dir_data = load_json('settings/dirs_files.json', os.getcwd())['dir_data']
ids = load_json('settings/ids.json', os.getcwd())

for i_subj, subj in enumerate(ids):
    # read ERs
    erp_s, erp_t, _ = read_erp(subj, dir_data, decim=1, notch=True)
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

    save_pickle(subj + '_erp_t', dir_save, np.mean(erp_t_filt, axis=0))
    save_pickle(subj + '_erp_s', dir_save, np.mean(erp_s_filt, axis=0))
    save_pickle(subj + '_env_t', dir_save, np.mean(env_t, axis=0))
    save_pickle(subj + '_env_s', dir_save, np.mean(env_s, axis=0))

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# collect all the data into single arrays
erp_t_all, _ = list_from_many(ids, dir_save, '_erp_t', 'pickle')
erp_s_all, _ = list_from_many(ids, dir_save, '_erp_s', 'pickle')
env_t_all, _ = list_from_many(ids, dir_save, '_env_t', 'pickle')
env_s_all, _ = list_from_many(ids, dir_save, '_env_s', 'pickle')

save_pickle('avg_erp_t', dir_save, erp_t_all)
save_pickle('avg_erp_s', dir_save, erp_s_all)
save_pickle('avg_env_t', dir_save, env_t_all)
save_pickle('avg_env_s', dir_save, env_s_all)
