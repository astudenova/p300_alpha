"""
This pipeline performs source reconstruction on stimulus data, and stores reconstructed ERs and
alpha rhythm envelopes.
Similar to p_read_erp_alpha_save.py but for source space.
"""
import os
import os.path as op
import numpy as np
import mne
from tools_general import load_json, load_pickle, save_pickle
from tools_lifedataset import read_erp, create_erp_for_source_reconstruction
from tools_signal import create_noise_cov, from_cont_to_epoch, compute_envelope, \
    filter_in_low_frequency

dir_save = load_json('settings/dirs_files.json', os.getcwd())['dir_save']
dir_data = load_json('settings/dirs_files.json', os.getcwd())['dir_data']
ids = load_json('settings/ids.json', os.getcwd())
alpha_peaks = load_pickle('settings/alpha_peaks.pkl', os.getcwd())

subjects_dir = load_json('settings/dirs_files.json', os.getcwd())['subjects_dir']
subject = 'fsaverage'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct6' + '-fwd.fif')
forward = mne.read_forward_solution(fwd_dir)

for i_subj, subj in enumerate(ids):
    erp_s, erp_t, _ = read_erp(subj, dir_data, notch=True, h_freq=20)

    decim = 10
    fs = erp_t.info['sfreq'] / decim
    alpha_pk = np.mean(alpha_peaks[i_subj])
    n_ch = len(erp_t.ch_names)

    # create an evoked array for further processing
    evoked_erp_t = create_erp_for_source_reconstruction(erp_t.copy(),
                                                        decim=decim, n_ch=n_ch)
    evoked_erp_s = create_erp_for_source_reconstruction(erp_s.copy(),
                                                        decim=decim, n_ch=n_ch)

    # for target stimulus
    # create noise covariance with a bias of data length
    noise_cov = create_noise_cov(evoked_erp_t.data.shape, evoked_erp_t.info)
    # source reconstruction with eLORETA
    
    inv_op = mne.minimum_norm.make_inverse_operator(evoked_erp_t.info, forward, noise_cov,
                                                    loose=1.0, fixed=False)
    stc_el = mne.minimum_norm.apply_inverse(evoked_erp_t.copy(), inverse_operator=inv_op,
                                            lambda2=0.05, method='eLORETA', pick_ori='normal')
    print('eLORETA fit is completed.')
    stc_data_epoched = from_cont_to_epoch(stc_el.data, n_epoch=len(erp_t.events),
                                          n_times=int(len(stc_el.times) / len(erp_t.events)))

    # compute averaged envelope for each voxel
    stc_t_env = compute_envelope(stc_data_epoched, len(erp_t.events), stc_el.shape[0], fs=fs,
                                 alpha_peaks=np.tile(alpha_pk, stc_el.shape[0]), subtract=True)
    stc_t_env_avg = np.mean(stc_t_env, axis=0)

    # compute average ER for each voxel
    stc_t_filt = filter_in_low_frequency(stc_data_epoched, fs, padlen=None)
    stc_t_avg = np.mean(stc_t_filt, axis=0)

    # for standard stimulus
    # create noise covariance with a bias of data length
    noise_cov = create_noise_cov(evoked_erp_s.data.shape, evoked_erp_s.info)
    # source reconstruction with eLORETA
    inv_op = mne.minimum_norm.make_inverse_operator(evoked_erp_s.info, forward, noise_cov,
                                                    loose=1.0, fixed=False)
    stc_el = mne.minimum_norm.apply_inverse(evoked_erp_s.copy(), inverse_operator=inv_op,
                                            lambda2=0.05, method='eLORETA', pick_ori='normal')
    print('eLORETA fit is completed.')
    stc_data_epoched = from_cont_to_epoch(stc_el.data, n_epoch=len(erp_s.events),
                                          n_times=int(len(stc_el.times) / len(erp_s.events)))

    # compute averaged envelope for each voxel
    stc_s_env = compute_envelope(stc_data_epoched, len(erp_s.events), stc_el.shape[0], fs=fs,
                                 alpha_peaks=np.tile(alpha_pk, stc_el.shape[0]), subtract=True)
    stc_s_env_avg = np.mean(stc_s_env, axis=0)

    # compute average ER for each voxel
    stc_s_filt = filter_in_low_frequency(stc_data_epoched, fs, padlen=None)
    stc_s_avg = np.mean(stc_s_filt, axis=0)

    # Check whether the specified path exists or not
    path = op.join(dir_save, 'eL_p300_alpha')
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory if it does not exist
        os.makedirs(path)

    # save
    save_pickle(subj + '_t', op.join(dir_save, 'eL_p300_alpha'), stc_t_avg)
    save_pickle(subj + '_t_env', op.join(dir_save, 'eL_p300_alpha'), stc_t_env_avg)
    save_pickle(subj + '_s', op.join(dir_save, 'eL_p300_alpha'), stc_s_avg)
    save_pickle(subj + '_s_env', op.join(dir_save, 'eL_p300_alpha'), stc_s_env_avg)

    print('ER and envelopes are saved.')

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')
