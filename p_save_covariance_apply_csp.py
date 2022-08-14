import os
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from tools_external import compute_ged, compute_patterns
from tools_general import save_pickle, load_pickle, load_json, list_from_many
from tools_lifedataset import read_erp
from tools_signal import from_epoch_to_cont, from_cont_to_epoch, apply_spatial_filter, pk_latencies_amplitudes

dir_raw = '/data/pt_02035/Data/eeg_data'

dir_save = '/data/p_02581/save'

ids = load_json('ids', os.getcwd())
alpha_peaks = load_pickle('alpha_peaks', os.getcwd())

# Step 1. Save covariances
for i_subj, subj in enumerate(ids):

    erp_s, erp_t, _ = read_erp(subj, decim=1, notch=True)

    fs = erp_t.info['sfreq']
    erp_s = erp_s.get_data(picks='eeg')
    erp_t = erp_t.get_data(picks='eeg')
    (n_epoch, n_sen, n_times) = erp_t.shape
    erp_times = erp_t.times
    win_post = np.array([0.3, 0.7])
    win_post_samples = np.array(
        [np.argmin(np.abs(erp_times - win_post[0])), np.argmin(np.abs(erp_times - win_post[1]))])

    # flatten epochs
    erp_t_all = from_epoch_to_cont(erp_t, n_sen, n_epoch)
    erp_s_all = from_epoch_to_cont(erp_s, n_sen, n_epoch)

    # filter around the peak
    alpha_peak = np.mean(alpha_peaks[i_subj])  # average over channels
    adj_band = np.array([alpha_peak - 2, alpha_peak + 2])
    b10, a10 = butter(N=2, Wn=adj_band / fs * 2, btype='bandpass')
    alpha_t_all = filtfilt(b10, a10, erp_t_all, padlen=500, axis=1)
    alpha_s_all = filtfilt(b10, a10, erp_s_all, padlen=500, axis=1)
    alpha_t_all_epoch = from_cont_to_epoch(alpha_t_all, n_epoch, n_times)
    alpha_s_all_epoch = from_cont_to_epoch(alpha_s_all, n_epoch, n_times)

    # compute covariance of each epoch
    cov_t = []
    cov_s = []
    for i in range(n_epoch):
        cov_t.append(np.real(np.cov(alpha_t_all_epoch[i, :n_sen, win_post_samples[0]:win_post_samples[1]])))
        cov_s.append(np.real(np.cov(alpha_s_all_epoch[i, :n_sen, win_post_samples[0]:win_post_samples[1]])))

    # save averaged over epochs covariances
    save_pickle(subj + '_cov_t', dir_save, np.mean(cov_t, axis=0))
    save_pickle(subj + '_cov_s', dir_save, np.mean(cov_s, axis=0))

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# Step 2. Compute csp from averaged-over-subjects covariance
# collect covariances into single array
cov_t_all, _ = list_from_many(ids, dir_save, '_cov_t', 'pickle')
cov_s_all, _ = list_from_many(ids, dir_save, '_cov_s', 'pickle')

# compute csp only for selected subjects
full_mask = load_pickle('full_mask', os.getcwd())
cov_mat_t_avg = np.mean(cov_t_all[full_mask], axis=0)  # average over subjects
cov_mat_s_avg = np.mean(cov_s_all[full_mask], axis=0)

csp_filter = compute_ged(cov_mat_s_avg, cov_mat_t_avg)
csp_pattern = compute_patterns(cov_mat_s_avg, csp_filter)
save_pickle('csp_filter', os.getcwd(), csp_filter)
save_pickle('csp_pattern', os.getcwd(), csp_pattern)
#
# csp_filter = load_pickle('csp_filter', os.getcwd())
# csp_pattern = load_pickle('csp_pattern', os.getcwd())

# Step 3. Apply csp on the data of each subject tp retrieve peak latency and peak amplitude of alpha amplitude
for i_subj, subj in enumerate(ids):
    _, erp_t, _ = read_erp(subj, dir_raw, decim=1, notch=True)

    fs = erp_t.info['sfreq']
    erp_t = erp_t.get_data(picks='eeg')
    (n_epoch, n_sen, n_times) = erp_t.shape

    # flatten epochs
    erp_t_all = from_epoch_to_cont(erp_t, n_sen, n_epoch)
    # filter around alpha peak
    alpha_peak = np.mean(alpha_peaks[i_subj])  # average over channels
    adj_band = np.array([alpha_peak - 2, alpha_peak + 2])
    b10, a10 = butter(N=2, Wn=adj_band / fs * 2, btype='bandpass')
    alpha_t_all = filtfilt(b10, a10, erp_t_all, padlen=500, axis=1)
    alpha_t_all_epoch = from_cont_to_epoch(alpha_t_all, n_epoch, n_times)

    # apply precomputed filter
    alpha_t_spat = apply_spatial_filter(alpha_t_all_epoch, csp_filter, csp_pattern, n_sen, n_epoch)
    alpha_t_spat_flat = from_epoch_to_cont(alpha_t_spat[:, np.newaxis, :], n_sen=1, n_epoch=n_epoch)
    env_t_spat = from_cont_to_epoch(np.abs(hilbert(alpha_t_spat_flat, axis=1)), n_epoch, n_times)
    # compute peak latency and peak amplitude from averaged time course
    env_t_spat_avg = np.mean(env_t_spat, axis=0).reshape((-1))
    env_t_peak = pk_latencies_amplitudes(env_t_spat_avg, np.array([.2, 1]), erp_t.times, direction='neg')[1:]
    save_pickle(subj + '_env_peak', dir_save, env_t_peak)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')

# collect all values into a single array
env_peak_all, _ = list_from_many(ids, dir_save, '_env_peak', 'pickle')
save_pickle('csp_env_peak_lat', '/data/p_02581', env_peak_all[:, 0])
save_pickle('csp_env_peak_amp', '/data/p_02581', env_peak_all[:, 1])
