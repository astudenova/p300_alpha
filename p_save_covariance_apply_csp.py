"""
This pipeline saves covariance matrices for each participant for both target and standard stimuli.
"""
import os
import numpy as np
from scipy.signal import hilbert
from tools_external import compute_ged, compute_patterns
from tools_general import save_pickle, load_pickle, load_json, list_from_many
from tools_lifedataset import read_erp
from tools_signal import from_epoch_to_cont, from_cont_to_epoch, apply_spatial_filter, \
    pk_latencies_amplitudes, filter_in_alpha_band

dir_save = os.path.join(load_json('settings/dirs_files.json', os.getcwd())['dir_save'],'erd/')
dir_data = load_json('settings/dirs_files.json', os.getcwd())['dir_data']
ids = load_json('settings/ids.json', os.getcwd())
alpha_peaks = load_pickle('settings/alpha_peaks.pkl', os.getcwd())

# Step 1. Save covariances
for i_subj, subj in enumerate(ids):

    erp_s, erp_t, _ = read_erp(subj, dir_data, decim=1, notch=True)

    fs = erp_t.info['sfreq']
    erp_times = erp_t.times
    erp_s = erp_s.get_data(picks='eeg')
    erp_t = erp_t.get_data(picks='eeg')
    (n_epoch, n_ch, n_times) = erp_t.shape
    win = np.array([0.3, 0.7])
    win_samples = np.array(
        [np.argmin(np.abs(erp_times - win[0])), np.argmin(np.abs(erp_times - win[1]))])

    # flatten epochs
    erp_t_all = from_epoch_to_cont(erp_t, n_ch, n_epoch)
    erp_s_all = from_epoch_to_cont(erp_s, n_ch, erp_s.shape[0])

    # filter around the peak
    alpha_peak = np.mean(alpha_peaks[i_subj])  # average over channels
    alpha_t_all = filter_in_alpha_band(erp_t_all, fs, padlen=500, alpha_peak=alpha_peak)
    alpha_s_all = filter_in_alpha_band(erp_s_all, fs, padlen=500, alpha_peak=alpha_peak)
    alpha_t_all_epoch = from_cont_to_epoch(alpha_t_all, n_epoch, n_times)
    alpha_s_all_epoch = from_cont_to_epoch(alpha_s_all, erp_s.shape[0], n_times)

    # compute covariance of each epoch
    cov_t = []
    cov_s = []
    for i in range(n_epoch):
        cov_t.append(np.real(np.cov(alpha_t_all_epoch[i, :n_ch, win_samples[0]:win_samples[1]])))
        cov_s.append(np.real(np.cov(alpha_s_all_epoch[i, :n_ch, win_samples[0]:win_samples[1]])))

    # save averaged over epochs covariances
    save_pickle(subj + '_cov_t', dir_save, np.mean(cov_t, axis=0))
    save_pickle(subj + '_cov_s', dir_save, np.mean(cov_s, axis=0))

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# Step 2. Compute csp from averaged-over-subjects covariance
# collect covariances into single array
cov_t_all, _ = list_from_many(ids, dir_save, '_cov_t', 'pickle')
cov_s_all, _ = list_from_many(ids, dir_save, '_cov_s', 'pickle')

# compute csp only for selected subjects
full_mask = load_pickle('settings/full_mask.pkl', os.getcwd())
cov_mat_t_avg = np.mean(cov_t_all[full_mask], axis=0)  # average over subjects
cov_mat_s_avg = np.mean(cov_s_all[full_mask], axis=0)

csp_filter = compute_ged(cov_mat_s_avg, cov_mat_t_avg)
csp_pattern = compute_patterns(cov_mat_s_avg, csp_filter)
save_pickle('csp_filter', os.getcwd(), csp_filter)
save_pickle('csp_pattern', os.getcwd(), csp_pattern)

csp_filter = load_pickle('csp_filter.pkl', os.getcwd())
csp_pattern = load_pickle('csp_pattern.pkl', os.getcwd())

# Step 3. Apply csp on the data of each subject
# to retrieve peak latency and peak amplitude of alpha amplitude
for i_subj, subj in enumerate(ids):
    _, erp_t, _ = read_erp(subj, dir_data, decim=1, notch=True)

    fs = erp_t.info['sfreq']
    erp_times = erp_t.times
    erp_t = erp_t.get_data(picks='eeg')
    (n_epoch, n_ch, n_times) = erp_t.shape

    # flatten epochs
    erp_t_all = from_epoch_to_cont(erp_t, n_ch, n_epoch)
    # filter around alpha peak
    alpha_peak = np.mean(alpha_peaks[i_subj])  # average over channels
    alpha_t_all = filter_in_alpha_band(erp_t_all, fs, padlen=500, alpha_peak=alpha_peak)
    alpha_t_all_epoch = from_cont_to_epoch(alpha_t_all, n_epoch, n_times)

    # apply precomputed filter
    alpha_t_spat = apply_spatial_filter(alpha_t_all_epoch, csp_filter, csp_pattern, n_ch, n_epoch)
    alpha_t_spat_flat = from_epoch_to_cont(alpha_t_spat[:, np.newaxis, :], n_ch=1, n_epoch=n_epoch)
    env_t_spat = from_cont_to_epoch(np.abs(hilbert(alpha_t_spat_flat, axis=1)), n_epoch, n_times)
    # compute peak latency and peak amplitude from averaged time course
    env_t_spat_avg = np.mean(env_t_spat, axis=0).reshape((-1))
    env_t_peak = pk_latencies_amplitudes(env_t_spat_avg, np.array([.2, 1]), erp_times,
                                         direction='neg')[0]
    erd = np.mean(env_t_spat_avg[int(env_t_peak[0])-50:int(env_t_peak[0])+50])/np.mean(env_t_spat_avg[150:250]) - 1
    save_pickle(subj + '_env_peak', dir_save, env_t_peak[1:])
    save_pickle(subj+'_erd',dir_save, erd)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# collect all values into a single array
env_peak_all, _ = list_from_many(ids, dir_save, '_env_peak', 'pickle')
erd_all, _ = list_from_many(ids,dir_save,'_erd','pickle')
save_pickle('csp_env_peak_lat', dir_save, env_peak_all[:, 0])
save_pickle('csp_env_peak_amp', dir_save, env_peak_all[:, 1])
save_pickle('csp_erd',dir_save,erd_all)
