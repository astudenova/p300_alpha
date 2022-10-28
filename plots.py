"""
This code plots the figures in Results.
"""
import os
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test, ttest_ind_no_p
from scipy.stats import pearsonr, ttest_rel, t, spearmanr
from pingouin import partial_corr
from tools_general import list_from_many, load_json_to_numpy, load_pickle, load_json, \
    scaler_transform, save_pickle, permutation_test_outcome
from tools_lifedataset import read_medications, composite_attention, composite_memory, \
    composite_executive, read_age, read_gender
from tools_plotting import plot_with_sem_one_line, topoplot_with_colorbar, parula_map, \
    plot_with_sem_color, annotated_heatmap, parula_map_backward, plot_brain_views
from tools_signal import pk_latencies_amplitudes, apply_spatial_filter, lda_

mpl.use("Qt5Agg")

dir_codes = os.getcwd()
dir_derr = load_json('settings_real/dirs_files.json', os.getcwd())['dir_derr']
ids_all = load_json('settings_real/ids.json', dir_codes)
erp_times = np.array(load_json('erp_times.json', dir_codes))
raw_info = load_pickle('raw_info.pkl', dir_codes)
reject_spec = load_json('reject_spec.json', dir_codes)
meds_ids, _ = read_medications(ids_all)
reject_ids = list(set(reject_spec + list(meds_ids)))
id_mask = np.intersect1d(ids_all, reject_ids, return_indices=True)[1]
ids = np.delete(ids_all, id_mask)
num_subj = len(ids)
n_ch = 31
age, age_ids = read_age(ids)
gender, _ = read_gender(ids)

# for source reconstruction
subjects_dir = load_json('settings_real/dirs_files.json', os.getcwd())['subjects_dir']

subject = 'fsaverage'

erp_times_dec = load_json_to_numpy('erp_times_dec.json', dir_codes)
stc_fixed = load_pickle('stc_fixed.pkl', dir_codes)

full_mask = np.logical_not(np.in1d(np.arange(0, len(ids_all)), id_mask))
# These files are generated with the script p_read_erp_alpha_save.py
avg_erp_t = load_pickle('avg_erp_t', dir_derr)[full_mask]
avg_erp_s = load_pickle('avg_erp_s', dir_derr)[full_mask]
avg_env_t = load_pickle('avg_env_t', dir_derr)[full_mask]
avg_env_s = load_pickle('avg_env_s', dir_derr)[full_mask]

pz_idx = np.where(np.array(raw_info.ch_names) == 'Pz')[0][0]

# ---------------------------------------------------------------
# FIGURE 2a
# ---------------------------------------------------------------

fig, ax = plt.subplots(1, 2, sharex=True)
ax = ax.flatten()
plot_with_sem_one_line(erp_times, avg_erp_t[:, pz_idx, :], ax, 0, xlim=[-0.2, 1.1],
                       ylim=[-1 * 10 ** (-6), 3.5 * 10 ** (-6)],
                       color_y='darkblue', color_y_sem='skyblue', label_y='target')
plot_with_sem_one_line(erp_times, avg_erp_s[:, pz_idx, :], ax, 0, xlim=[-0.2, 1.1],
                       ylim=[-1 * 10 ** (-6), 3.5 * 10 ** (-6)],
                       color_y='steelblue', color_y_sem='powderblue', label_y='standard')
plot_with_sem_one_line(erp_times, avg_env_t[:, pz_idx, :], ax, 1, xlim=[-0.2, 1.1],
                       ylim=[1 * 10 ** (-6), 4.5 * 10 ** (-6)],
                       color_y='orange', color_y_sem='moccasin', label_y='target')
plot_with_sem_one_line(erp_times, avg_env_s[:, pz_idx, :], ax, 1, xlim=[-0.2, 1.1],
                       ylim=[1 * 10 ** (-6), 4.5 * 10 ** (-6)],
                       color_y='tan', color_y_sem='antiquewhite', label_y='standard')

# ---------------------------------------------------------------
# FIGURE 2b
# ---------------------------------------------------------------

corr_t = np.array([pearsonr(np.mean(avg_erp_t, axis=0)[i], np.mean(avg_env_t, axis=0)[i]) for i in range(n_ch)])

thr_sen = 10 ** (-4) / n_ch
topoplot_with_colorbar(corr_t[:,0], raw_info, cmap=parula_map(),
                       mask=corr_t[:,1] < thr_sen, vmin=-1, vmax=1)

# ---------------------------------------------------------------
# FIGURE 3
# ---------------------------------------------------------------

erp_peaks_avg = []
for i_subj, subj in enumerate(ids):
    avg_erp_pk = pk_latencies_amplitudes(avg_erp_t[i_subj, pz_idx],
                                         np.array([0.2, 1]), erp_times, direction='pos')[0][1]
    if avg_erp_pk != 0:
        erp_peaks_avg.append(avg_erp_pk)
    else:
        erp_peaks_avg.append(.5)
        print('ERP peak is not found for subject ' + str(subj))
        print('Setting the peak latency to 0.5.')

erd_avg = np.zeros((len(ids), n_ch))
for i_subj in range(len(ids)):
    pk_sample_erp = np.argmin(np.abs(erp_times - erp_peaks_avg[i_subj]))
    erd_avg[i_subj] = 1 - (np.mean(avg_env_t[i_subj, :, pk_sample_erp - 50:pk_sample_erp + 50], axis=1) / np.mean(
        avg_env_t[i_subj, :, 150:250], axis=1))

erd_pz = erd_avg[:, pz_idx]

erd_bins = list(np.percentile(erd_pz, np.arange(0, 100, 20)))
erd_bins.append(np.max(erd_pz))
erd_bins = np.array(erd_bins)
binned_idx = np.zeros(len(erd_pz), )
for ai in range(5):
    binned_idx = binned_idx + ai * ((erd_bins[ai] <= erd_pz) * (erd_pz < erd_bins[ai + 1]))

# ---------------------------------------------------------------
# FIGURE 3ab
# ---------------------------------------------------------------

fig, ax = plt.subplots(1, 2)
color_bins_erp = ['#00406B', '#104E8B', '#1874CD', '#1C86EE', '#1EA1FF']
color_bins_env = ['#3B1E00', '#633200', '#8B4500', '#B35900', '#E57600']
label_bsi = ['ERD-bin1: ' + str("{:.2f}".format(erd_bins[0])) + ' - ' + str("{:.2f}".format(erd_bins[1])),
             'ERD-bin2: ' + str("{:.2f}".format(erd_bins[1])) + ' - ' + str("{:.2f}".format(erd_bins[2])),
             'ERD-bin3: ' + str("{:.2f}".format(erd_bins[2])) + ' - ' + str("{:.2f}".format(erd_bins[3])),
             'ERD-bin4: ' + str("{:.2f}".format(erd_bins[3])) + ' - ' + str("{:.2f}".format(erd_bins[4])),
             'ERD-bin5: ' + str("{:.2f}".format(erd_bins[4])) + ' - ' + str(
                 "{:.2f}".format(np.max(erd_bins)))]
plot_with_sem_color(erp_times, [avg_erp_t[binned_idx == 0, pz_idx, :], avg_erp_t[binned_idx == 1, pz_idx, :],
                                avg_erp_t[binned_idx == 2, pz_idx, :], avg_erp_t[binned_idx == 3, pz_idx, :],
                                avg_erp_t[binned_idx == 4, pz_idx, :]],
                    ax, 0, xlim=[-.3, 1.25],
                    color_y=color_bins_erp,
                    color_y_sem=color_bins_erp,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=label_bsi)
ax[0].set_title('ERP')
ax[0].set_xlim([-0.2, 1.1])
plot_with_sem_color(erp_times, [avg_env_t[binned_idx == 0, pz_idx, :], avg_env_t[binned_idx == 1, pz_idx, :],
                                avg_env_t[binned_idx == 2, pz_idx, :], avg_env_t[binned_idx == 3, pz_idx, :],
                                avg_env_t[binned_idx == 4, pz_idx, :]],
                    ax, 1, xlim=[-.3, 1.25],
                    color_y=color_bins_env,
                    color_y_sem=color_bins_env,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=label_bsi)
ax[1].set_title('alpha')
ax[1].set_xlim([-0.2, 1.1])

# ---------------------------------------------------------------
# FIGURE 3c
# ---------------------------------------------------------------
sen_sorted = load_json('sen_sorted_idx.json', dir_codes)
# make bins for each channel
idx_bin = np.zeros((n_ch, 5, len(ids)), dtype=bool)
for ch in range(n_ch):
    erd_bins = np.percentile(np.sort(erd_avg[:, ch]), np.arange(0, 100, 20))
    # create binning idx
    idx_bin[ch, 0] = erd_avg[:, ch] < erd_bins[1]
    idx_bin[ch, -1] = erd_avg[:, ch] > erd_bins[-1]
    for i in range(1, 5 - 1):
        idx_bin[ch, i] = np.multiply(erd_avg[:, ch] > erd_bins[i], erd_avg[:, ch] < erd_bins[i + 1])

# arrange P300 according to bins
ampl_full_t = np.zeros((2, len(ids), n_ch, len(erp_times)))
idx_bin1 = idx_bin[:, 0]
idx_bin5 = idx_bin[:, 4]
for i_subj in range(len(ids)):
    subj_mask_bin1 = idx_bin1[:, i_subj]
    subj_mask_bin5 = idx_bin5[:, i_subj]
    ampl_full_t[0, i_subj] = np.multiply(avg_erp_t[i_subj, :n_ch].T, subj_mask_bin1).T
    ampl_full_t[1, i_subj] = np.multiply(avg_erp_t[i_subj, :n_ch].T, subj_mask_bin5).T

ampl_t = np.zeros((2, np.sum(idx_bin[0, 0]), n_ch, len(erp_times)))
for i in range(2):
    for i_ch in range(n_ch):
        cntr_t = 0
        for i_subj in range(len(ids)):
            if np.sum(ampl_full_t[i, i_subj, i_ch] == 0) != len(erp_times):
                ampl_t[i, cntr_t, i_ch] = ampl_full_t[i, i_subj, i_ch]
                cntr_t += 1

# permutation test
t_threshold = t.ppf(q=1 - 10 ** (-4) / 2, df=num_subj - 1)

adjacency, ch_names = find_ch_adjacency(raw_info, ch_type='eeg')
X = np.moveaxis(ampl_t, 3, 2)
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=10000,
                                             threshold=t_threshold, stat_fun=ttest_ind_no_p, tail=0,
                                             n_jobs=1, buffer_size=None, adjacency=adjacency)

F_obs, F_obs_sig = permutation_test_outcome(cluster_stats)

plt.imshow(F_obs[:, sen_sorted].T, aspect=15, cmap=parula_map())
plt.imshow(F_obs_sig[:, sen_sorted].T, aspect=15, cmap='Greys', alpha=0.6)

t450 = np.argmin(np.abs(erp_times - .45))
ch_mask = np.zeros((n_ch,), dtype=bool)
ch_mask[F_obs_sig[t450] != 1] = True
topoplot_with_colorbar(F_obs[t450], raw_info=raw_info,
                       cmap=parula_map(), mask=ch_mask, vmin=np.min(F_obs), vmax=np.max(F_obs))

# ---------------------------------------------------------------
# FIGURE 4
# ---------------------------------------------------------------
# build topographies from avg peak values
erp_topo_avg = np.zeros((len(ids), n_ch))
env_topo_avg = np.zeros((len(ids), n_ch))
noerp_topo_avg = np.zeros((len(ids), n_ch))
noenv_topo_avg = np.zeros((len(ids), n_ch))
for i_subj in range(len(ids)):
    pk_sample_erp = np.argmin(np.abs(erp_times - erp_peaks_avg[i_subj]))
    env_topo_avg[i_subj] = np.mean(avg_env_t[i_subj, :, pk_sample_erp - 50:pk_sample_erp + 50], axis=1)
    noenv_topo_avg[i_subj] = np.mean(avg_env_s[i_subj, :, pk_sample_erp - 50:pk_sample_erp + 50], axis=1)
    erp_topo_avg[i_subj] = avg_erp_t[i_subj, :n_ch, pk_sample_erp]
    noerp_topo_avg[i_subj] = avg_erp_s[i_subj, :n_ch, pk_sample_erp]

topoplot_with_colorbar(np.mean(erp_topo_avg - noerp_topo_avg, axis=0),
                       raw_info, cmap=parula_map())
topoplot_with_colorbar(np.mean(env_topo_avg / noenv_topo_avg, axis=0),
                       raw_info, cmap=parula_map(), vmin=0.55, vmax=0.80)

print('Spearman correlation: ' +
      str(spearmanr(np.mean(erp_topo_avg - noerp_topo_avg, axis=0), np.mean(env_topo_avg / noenv_topo_avg, axis=0))))

# ---------------------------------------------------------------
# FIGURE 5a
# ---------------------------------------------------------------

# This file is generated with the script p_compute_bsi.py
bsi_all = load_pickle('bsi_all', dir_derr)[full_mask]
bsi_mean = np.mean(bsi_all, axis=0)
topoplot_with_colorbar(bsi_mean, raw_info, cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 5c
# ---------------------------------------------------------------

color_bins = ['#00406B', '#104E8B', '#1874CD', '#1C86EE', '#1EA1FF']
bsi_bins = np.percentile(np.sort(bsi_all[:, pz_idx]), np.arange(0, 100, 20))
# create binning idx
idx_bin = np.zeros((5, len(ids)), dtype=bool)
idx_bin[0] = bsi_all[:, pz_idx] < bsi_bins[1]
idx_bin[-1] = bsi_all[:, pz_idx] > bsi_bins[-1]
for i in range(1, 5 - 1):
    idx_bin[i] = np.multiply(bsi_all[:, pz_idx] > bsi_bins[i], bsi_all[:, pz_idx] < bsi_bins[i + 1])
# over 5 bins
fig, ax = plt.subplots(1, 1)
label_bsi = ['BSI-bin1: ' + str("{:.2f}".format(bsi_bins[0])) + ' - ' + str("{:.2f}".format(bsi_bins[1])),
             'BSI-bin2: ' + str("{:.2f}".format(bsi_bins[1])) + ' - ' + str("{:.2f}".format(bsi_bins[2])),
             'BSI-bin3: ' + str("{:.2f}".format(bsi_bins[2])) + ' - ' + str("{:.2f}".format(bsi_bins[3])),
             'BSI-bin4: ' + str("{:.2f}".format(bsi_bins[3])) + ' - ' + str("{:.2f}".format(bsi_bins[4])),
             'BSI-bin5: ' + str("{:.2f}".format(bsi_bins[4])) + ' - ' + str(
                 "{:.2f}".format(np.max(bsi_all[:, pz_idx])))]
plot_with_sem_color(erp_times, [avg_erp_t[idx_bin[0], pz_idx, :], avg_erp_t[idx_bin[1], pz_idx, :],
                                avg_erp_t[idx_bin[2], pz_idx, :], avg_erp_t[idx_bin[3], pz_idx, :],
                                avg_erp_t[idx_bin[4], pz_idx, :]],
                    ax, None, xlim=[-.3, 1.25],
                    color_y=color_bins,
                    color_y_sem=color_bins,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=label_bsi)
ax.set_title('ERP')
ax.set_xlim([-0.2, 1.1])

# ---------------------------------------------------------------
# FIGURE 5b
# ---------------------------------------------------------------

fig, ax = plt.subplots()
N, bins, patches = ax.hist(bsi_all[:, pz_idx], bins=50, color='dodgerblue')
bsi_color = np.array([np.hstack((bsi_bins[1:], np.max(bsi_all[:, pz_idx]))),
                      color_bins])
for patch in patches:
    if patch.xy[0] < np.float(bsi_color[0, 0]):
        patch.set_facecolor(bsi_color[1, 0])
    elif patch.xy[0] < np.float(bsi_color[0, 1]):
        patch.set_facecolor(bsi_color[1, 1])
    elif patch.xy[0] < np.float(bsi_color[0, 2]):
        patch.set_facecolor(bsi_color[1, 2])
    elif patch.xy[0] < np.float(bsi_color[0, 3]):
        patch.set_facecolor(bsi_color[1, 3])
    elif patch.xy[0] < np.float(bsi_color[0, 4]):
        patch.set_facecolor(bsi_color[1, 4])

# ---------------------------------------------------------------
# FIGURE 5d
# ---------------------------------------------------------------

sen_sorted = load_json('sen_sorted_idx.json', dir_codes)
# make bins for each channel
idx_bin = np.zeros((n_ch, 5, len(ids)), dtype=bool)
for ch in range(n_ch):
    bsi_bins = np.percentile(np.sort(bsi_all[:, ch]), np.arange(0, 100, 20))
    # create binning idx
    idx_bin[ch, 0] = bsi_all[:, ch] < bsi_bins[1]
    idx_bin[ch, -1] = bsi_all[:, ch] > bsi_bins[-1]
    for i in range(1, 5 - 1):
        idx_bin[ch, i] = np.multiply(bsi_all[:, ch] > bsi_bins[i], bsi_all[:, ch] < bsi_bins[i + 1])

# arrange P300 according to bins
ampl_full_t = np.zeros((2, len(ids), n_ch, len(erp_times)))
idx_bin1 = idx_bin[:, 0]
idx_bin5 = idx_bin[:, 4]
for i_subj in range(len(ids)):
    subj_mask_bin1 = idx_bin1[:, i_subj]
    subj_mask_bin5 = idx_bin5[:, i_subj]
    ampl_full_t[0, i_subj] = np.multiply(avg_erp_t[i_subj, :n_ch].T, subj_mask_bin1).T
    ampl_full_t[1, i_subj] = np.multiply(avg_erp_t[i_subj, :n_ch].T, subj_mask_bin5).T

ampl_t = np.zeros((2, np.sum(idx_bin[0, 0]), n_ch, len(erp_times)))
for i in range(2):
    for i_ch in range(n_ch):
        cntr_t = 0
        for i_subj in range(len(ids)):
            if np.sum(ampl_full_t[i, i_subj, i_ch] == 0) != len(erp_times):
                ampl_t[i, cntr_t, i_ch] = ampl_full_t[i, i_subj, i_ch]
                cntr_t += 1

# permutation test
t_threshold = t.ppf(q=1 - 10 ** (-4) / 2, df=num_subj - 1)

adjacency, ch_names = find_ch_adjacency(raw_info, ch_type='eeg')
X = np.moveaxis(ampl_t, 3, 2)
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=10000,
                                             threshold=t_threshold, stat_fun=ttest_ind_no_p, tail=0,
                                             n_jobs=1, buffer_size=None, adjacency=adjacency)

F_obs, F_obs_sig = permutation_test_outcome(cluster_stats)

plt.imshow(F_obs[:, sen_sorted].T, aspect=15, cmap=parula_map())
plt.imshow(F_obs_sig[:, sen_sorted].T, aspect=15, cmap='Greys', alpha=0.6)

# ---------------------------------------------------------------
# FIGURE 5e
# ---------------------------------------------------------------

t500 = np.argmin(np.abs(erp_times - .5))
ch_mask = np.zeros((n_ch,), dtype=bool)
ch_mask[F_obs_sig[t500] != 1] = True
topoplot_with_colorbar(F_obs[t500], raw_info=raw_info,
                       cmap=parula_map(), mask=ch_mask, vmin=np.min(F_obs), vmax=np.max(F_obs))

# ---------------------------------------------------------------
# FIGURE 6a
# ---------------------------------------------------------------
n_source = 8196
thr_source = t.ppf(q=1 - 10 ** (-4) / n_source / 2, df=num_subj - 1)

# These files are generated with the script p_source_reconstruction.py
stc_p300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t', 'pickle')
stc_p300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t_env', 'pickle')
stc_nop300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s', 'pickle')
stc_nop300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s_env', 'pickle')

win = np.array([0.3, 0.7])
win_samples = np.array([np.argmin(np.abs(erp_times_dec - win[0])),
                        np.argmin(np.abs(erp_times_dec - win[1]))])
X1 = np.mean(stc_p300[:, :, win_samples[0]:win_samples[1]], axis=2)
X2 = np.mean(stc_nop300[:, :, win_samples[0]:win_samples[1]], axis=2)

t_vox = [ttest_rel(X1[:, vi], X2[:, vi])[0] for vi in range(n_source)]
print(np.sum(np.abs(t_vox) > thr_source))

X_avg_diff = np.squeeze(np.mean(X1 - X2, axis=0))

data_to_plot = np.multiply(X_avg_diff, np.abs(t_vox) > thr_source)

clim = dict(kind='value', lims=[np.nanmin(X_avg_diff), np.nanmean(X_avg_diff), np.nanmax(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'p300', cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 6b
# ---------------------------------------------------------------

X1 = np.mean(stc_nop300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2) / \
     np.mean(stc_p300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2)
X2 = np.ones(X1.shape)

t_vox = [ttest_rel(X1[:, vi], X2[:, vi])[0] for vi in range(n_source)]

print(np.sum(np.array(t_vox) > thr_source))

X_avg_diff = np.squeeze(np.mean(X1 / X2, axis=0))

data_to_plot = np.multiply(X_avg_diff, np.array(t_vox) > thr_source)
clim = dict(kind='value', lims=[1, np.mean(X_avg_diff), np.max(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'alpha_env', cmap=parula_map_backward())

# ---------------------------------------------------------------
# FIGURE 7
# ---------------------------------------------------------------
# FIGURE 7b
# ---------------------------------------------------------------

lda_filter, lda_pattern = lda_(avg_erp_t[:, :n_ch], avg_erp_s[:, :n_ch], [0.3, 0.7], erp_times)
save_pickle('lda_filter', dir_derr, lda_filter)
save_pickle('lda_pattern', dir_derr, lda_pattern)
topoplot_with_colorbar(lda_pattern, raw_info, cmap=parula_map())

lda_erp_peak_lat = np.zeros((len(ids),))
lda_erp_peak_amp = np.zeros((len(ids),))
for i_subj, subj in enumerate(ids):
    erp_t = avg_erp_t[i_subj][:n_ch][np.newaxis, :, :]

    erp_t_spat = apply_spatial_filter(erp_t, lda_filter, lda_pattern, n_ch, n_epoch=1).reshape((-1))
    erp_t_peak = pk_latencies_amplitudes(erp_t_spat, np.array([.2, 1]),
                                         erp_times, direction='pos')[0][1:]

    lda_erp_peak_lat[i_subj] = erp_t_peak[0]
    lda_erp_peak_amp[i_subj] = erp_t_peak[1]

# ---------------------------------------------------------------
# FIGURE 7c
# ---------------------------------------------------------------

# These files are generated with the script p_save_covariance_apply_csp.py
csp_filter = load_pickle('csp_filter_real.pkl', os.getcwd())
csp_pattern = load_pickle('csp_pattern_real.pkl', os.getcwd())

topoplot_with_colorbar(csp_pattern, raw_info, cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 7a
# ---------------------------------------------------------------

attention_comp, attention_ids = composite_attention(ids)
memory_comp, memory_ids = composite_memory(ids)
executive_comp, executive_ids = composite_executive(ids)

# These files are generated with the script p_save_covariance_apply_csp.py
csp_env_peak_lat = load_pickle('csp_env_peak_lat', dir_derr)
csp_env_peak_amp = load_pickle('csp_env_peak_amp', dir_derr)

# spatially filtered
dv1 = lda_erp_peak_amp
dv2 = csp_env_peak_amp
dv3 = lda_erp_peak_lat
dv4 = csp_env_peak_lat

dv1_out = scaler_transform(dv1, scaler='standard')
dv2_out = scaler_transform(dv2, scaler='standard')
dv3_out = scaler_transform(dv3, scaler='standard')
dv4_out = scaler_transform(dv4, scaler='standard')

ids_attention_executive_ids, idx1, idx2 = np.intersect1d(executive_ids,
                                                         attention_ids, return_indices=True)
ids_attention_executive_memory_ids, idx3, idx4 = np.intersect1d(ids_attention_executive_ids,
                                                                memory_ids, return_indices=True)
_, idx5, idx6 = np.intersect1d(ids, ids_attention_executive_memory_ids, return_indices=True)

data_totest = pd.DataFrame()
data_totest['erpA'] = dv1_out[idx5]
data_totest['envA'] = dv2_out[idx5]
data_totest['erpL'] = dv3_out[idx5]
data_totest['envL'] = dv4_out[idx5]
data_totest['age'] = age[idx5]
data_totest['attention'] = attention_comp[idx2][idx3]
data_totest['executive'] = executive_comp[idx1][idx3]
data_totest['memory'] = memory_comp[idx4]

var_to_corr = ['erpA', 'envA', 'erpL', 'envL', 'age', 'attention', 'memory', 'executive']
corr_par = np.zeros((len(var_to_corr), len(var_to_corr)))
corr_par_pval = np.zeros((len(var_to_corr), len(var_to_corr)))
for i, var_i in enumerate(var_to_corr):
    for j, var_j in enumerate(var_to_corr):
        if i > j:
            try:
                corr_tmp_par = partial_corr(data_totest, var_i, var_j,
                                            covar='age', method='spearman')
                corr_par[i, j] = corr_par[j, i] = corr_tmp_par['r']
                corr_par_pval[i, j] = corr_par_pval[j, i] = corr_tmp_par['p-val']
            except:
                print('no age')

annotated_heatmap(var_to_corr[:4], var_to_corr[5:], np.round(corr_par[:4, 5:], 2).T,
                  cbarlabel='Spearman coefficient', cmap=parula_map())
annotated_heatmap(var_to_corr[:4], var_to_corr[5:], 1 * (corr_par_pval < (0.05 / 3))[:4, 5:].T,
                  cbarlabel='Spearman coefficient', cmap=parula_map())

# ---------------------------------------------------------------
# SUPPLEMENTARY MATERIAL
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Fig S3a
# ---------------------------------------------------------------

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True)
grand_avg_erp_t = np.mean(avg_erp_t, axis=0)
ax = ax.flatten()
for i in range(n_ch):
    ax[i].plot(erp_times, grand_avg_erp_t[i], c='darkblue', linewidth=3)
    ax[i].set_title(raw_info.ch_names[i], fontsize=6)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
ax[24].set_xlabel('Time, s')
ax[24].set_ylabel('Amplitude, uV')
ax[0].set_xlim([-0.2, 1.1])

# ---------------------------------------------------------------
# Fig S3b
# ---------------------------------------------------------------

grand_avg_env_t = np.mean(avg_env_t, axis=0)
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(n_ch):
    ax[i].plot(erp_times, grand_avg_env_t[i], c='orange', linewidth=3)
    ax[i].set_title(raw_info.ch_names[i], fontsize=6)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
ax[24].set_xlabel('Time, s')
ax[24].set_ylabel('Amplitude, uV')
ax[0].set_xlim([-0.2, 1.1])
