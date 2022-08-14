import mne
import os
import os.path as op
import matplotlib as mpl
import scipy

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import pearsonr

from tools_general import load_json, save_pickle, save_json
from tools_lifedataset import read_erp, create_erp_for_source_reconstruction
from tools_signal import create_noise_cov, from_cont_to_epoch, compute_envelope, from_epoch_to_cont

dir_raw = '/data/pt_02035/Data/eeg_data'

dir_save = '/data/p_02581/save'

ids = load_json('ids', '/data/hu_studenova/PycharmProjects/bsi_pipeline')
alpha_peaks = load_json('alpha_peaks', '/data/p_02581')

# for dipole fitting
subjects_dir = '/data/hu_studenova/mne_data/MNE-fsaverage-data'
subject = 'fsaverage'
_oct = '6'
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
trans_dir = op.join(subjects_dir, subject, 'bem', subject + '-trans.fif')
bem_sol_dir = op.join(subjects_dir, subject, 'bem', subject + '-5120-5120-5120-bem-sol.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-inv.fif')
forward = mne.read_forward_solution(fwd_dir)


for i_subj, subj in enumerate(ids):

    erp_s, erp_t, _ = read_erp(subj, notch=True, h_freq=20)

    decim = 10
    fs = erp_t.info['sfreq'] / decim
    alpha_pk = np.mean(alpha_peaks[i_subj])
    low = 3
    b_lf, a_lf = butter(N=4, Wn=low / fs * 2, btype='lowpass')

    # create an evoked array for further processing
    evoked_erp_t = create_erp_for_source_reconstruction(erp_t.copy(), decim=decim)
    evoked_erp_s = create_erp_for_source_reconstruction(erp_s.copy(), decim=decim)

    # for target stimulus
    # create noise covariance with a bias of data length
    noise_cov = create_noise_cov(evoked_erp_t.data.shape, evoked_erp_t.info)
    # source reconstruction with eLORETA
    inv_op = mne.minimum_norm.make_inverse_operator(evoked_erp_t.info, forward, noise_cov, loose=1.0, depth=5,
                                                    fixed=False)
    stc_el = mne.minimum_norm.apply_inverse(evoked_erp_t.copy(),
                                            inverse_operator=inv_op, lambda2=0.05, method='eLORETA', pick_ori='normal')
    print('eLORETA fit is completed.')
    stc_data_epoched = from_cont_to_epoch(stc_el.data, n_epoch=len(erp_t.events),
                                          n_times=int(len(stc_el.times) / len(erp_t.events)))

    # compute averaged envelope for each voxel
    stc_t_env = compute_envelope(stc_data_epoched, len(erp_t.events), stc_el.shape[0], fs=fs,
                                 alpha_peaks=np.tile(alpha_pk, stc_el.shape[0]), subtract=True)
    stc_t_env_avg = np.mean(stc_t_env, axis=0)

    # compute average ER for each voxel
    stc_t_filt = filtfilt(b_lf, a_lf, stc_data_epoched, axis=2)
    stc_t_avg = np.mean(stc_t_filt, axis=0)

    # for standard stimulus
    # create noise covariance with a bias of data length
    noise_cov = create_noise_cov(evoked_erp_s.data.shape, evoked_erp_s.info)
    # source reconstruction with eLORETA
    inv_op = mne.minimum_norm.make_inverse_operator(evoked_erp_s.info, forward, noise_cov, loose=1.0, depth=5,
                                                    fixed=False)
    stc_el = mne.minimum_norm.apply_inverse(evoked_erp_s.copy(),
                                            inverse_operator=inv_op, lambda2=0.05, method='eLORETA', pick_ori='normal')
    print('eLORETA fit is completed.')
    stc_data_epoched = from_cont_to_epoch(stc_el.data, n_epoch=len(erp_s.events),
                                          n_times=int(len(stc_el.times) / len(erp_s.events)))

    # compute averaged envelope for each voxel
    stc_s_env = compute_envelope(stc_data_epoched, len(erp_s.events), stc_el.shape[0], fs=fs,
                                 alpha_peaks=np.tile(alpha_pk, stc_el.shape[0]), subtract=True)
    stc_s_env_avg = np.mean(stc_s_env, axis=0)

    # compute average ER for each voxel
    stc_s_filt = filtfilt(b_lf, a_lf, stc_data_epoched)
    stc_s_avg = np.mean(stc_s_filt, axis=0)

    # save
    save_pickle(subj + '_t', dir_save, stc_t_avg)
    save_pickle(subj + '_t_env', dir_save, stc_t_env_avg)
    save_pickle(subj + '_s', dir_save, stc_s_avg)
    save_pickle(subj + '_s_env', dir_save, stc_s_env_avg)

    print('ER and envelopes are saved.')

    # correlation
    n_source = stc_el._data.shape[0]
    n_epoch = len(erp_t.events)
    n_times = int(len(stc_el.times) / len(erp_s.events)) - 20
    # target
    erp_t_filt_flat = from_epoch_to_cont(stc_t_filt, n_source, n_epoch, wings=10)
    env_t_flat = from_epoch_to_cont(stc_t_env, n_source, n_epoch, wings=10)
    corr_t = [pearsonr(erp_t_filt_flat[i], env_t_flat[i])[0] for i in range(n_source)]
    print('Correlation for target is computed.')

    # standard with subsample
    s_sample = np.random.random_integers(low=0, high=erp_s.events.shape[0] - 1, size=(n_epoch,))
    erp_s_filt_flat = from_epoch_to_cont(stc_s_filt[s_sample], n_source, n_epoch, wings=10)
    env_s_flat = from_epoch_to_cont(stc_s_env[s_sample], n_source, n_epoch, wings=10)
    corr_s = [pearsonr(erp_s_filt_flat[i], env_s_flat[i])[0] for i in range(n_source)]
    print('Correlation for standard is computed.')

    save_json(subj + '_corr_t', dir_save, corr_t)
    save_json(subj + '_corr_s', dir_save, corr_s)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')
