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
from scipy.stats import pearsonr, ttest_rel, t
from pingouin import partial_corr
from tools_general import list_from_many, load_json_to_numpy, load_pickle, load_json, \
    scaler_transform, save_pickle
from tools_lifedataset import read_medications, composite_attention, composite_memory, \
    composite_executive, read_age, read_gender
from tools_plotting import plot_with_sem_one_line, topoplot_with_colorbar, parula_map, \
    plot_with_sem_color, annotated_heatmap, parula_map_backward, plot_brain_views
from tools_signal import pk_latencies_amplitudes, apply_spatial_filter, lda_

mpl.use("Qt5Agg")

dir_codes = os.getcwd()
dir_derr = load_json('settings_real/dirs_files', os.getcwd())['dir_derr']
ids_all = load_json('settings_real/ids_real', dir_codes)
erp_times = np.array(load_json('erp_times', dir_codes))
raw_info = load_pickle('raw_info', dir_codes)
reject_spec = load_json('reject_spec', dir_codes)
meds_ids, _ = read_medications(ids_all)
reject_ids = list(set(reject_spec + list(meds_ids)))
id_mask = np.intersect1d(ids_all, reject_ids, return_indices=True)[1]
ids = np.delete(ids_all, id_mask)
num_subj = len(ids)
n_ch = 31
age, age_ids = read_age(ids)
gender, _ = read_gender(ids)

# for source reconstruction
subjects_dir = load_json('settings_real/dirs_files', os.getcwd())['subjects_dir']
subject = 'fsaverage'

erp_times_dec = load_json_to_numpy('erp_times_dec', dir_codes)
stc_fixed = load_pickle('stc_fixed', dir_codes)

full_mask = np.logical_not(np.in1d(np.arange(0, len(ids_all)), id_mask))
avg_erp_t = load_pickle('avg_erp_t', dir_derr)[full_mask]
avg_erp_s = load_pickle('avg_erp_s', dir_derr)[full_mask]
avg_env_t = load_pickle('avg_env_t', dir_derr)[full_mask]
avg_env_s = load_pickle('avg_env_s', dir_derr)[full_mask]

pz_idx = np.where(np.array(raw_info.ch_names) == 'Pz')[0][0]

# -----------------------------------------------------------
# FIGURE 1a
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# FIGURE 1b
# -----------------------------------------------------------

corr_t = load_pickle('corr_t', dir_derr)[full_mask]
corr_s = load_pickle('corr_s', dir_derr)[full_mask]
t_stat = [ttest_rel(corr_t[:, i], corr_s[:, i])[0] for i in range(n_ch)]

thr_sen = t.ppf(q=1 - 10 ** (-4) / n_ch / 2, df=num_subj - 1)
topoplot_with_colorbar(t_stat, raw_info, cmap=parula_map(),
                       mask=np.abs(t_stat) > thr_sen)

# ---------------------------------------------------------------
# FIGURE 2
# _______________________________________________________________

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

# -----------------------------------------------------------------
# FIGURE 3a
# -----------------------------------------------------------------

bsi_all = load_pickle('bsi_all', dir_derr)[full_mask]
bsi_mean = np.mean(bsi_all, axis=0)
topoplot_with_colorbar(bsi_mean, raw_info, cmap=parula_map())

# compute mode for later
bsi_mode = np.zeros((n_ch,))
for ch in range(n_ch):
    n, bins, _ = plt.hist(bsi_all[:, ch], bins=50)
    mode_index = np.argmax(n)
    bsi_mode[ch] = (bins[mode_index] + bins[mode_index + 1]) / 2

# -----------------------------------------------------------------
# FIGURE 3b
# -----------------------------------------------------------------

plt.hist(bsi_all[:, pz_idx], bins=50, color='dodgerblue')
plt.axvline(bsi_mean[pz_idx], color='k', linestyle='dotted', linewidth=3, label='mean')
plt.axvline(bsi_mode[pz_idx], color='k', linestyle='dashed', linewidth=3, label='mode')
plt.legend()
plt.xlabel('BSI')
plt.title('BSI at Pz')

# -----------------------------------------------------------------
# FIGURE 3c
# -----------------------------------------------------------------

corr_bsi_corr_t = np.zeros((n_ch,))
pval_bsi_corr_t = np.zeros((n_ch,))
for ch in range(n_ch):
    pearson_tmp = pearsonr(bsi_all[:, ch], corr_t[:, ch])
    corr_bsi_corr_t[ch] = pearson_tmp[0]
    pval_bsi_corr_t[ch] = pearson_tmp[1]

topoplot_with_colorbar(corr_bsi_corr_t, raw_info=raw_info,
                       cmap=parula_map(), mask=pval_bsi_corr_t < 0.0001 / n_ch)

# -----------------------------------------------------------------
# FIGURE 4a
# -----------------------------------------------------------------

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

# -----------------------------------------------------------------
# FIGURE 4b
# -----------------------------------------------------------------

sen_sorted = load_json('sen_sorted_idx', dir_codes)
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
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=t_threshold, stat_fun=ttest_ind_no_p, tail=0,
                                             n_jobs=1, buffer_size=None, adjacency=adjacency)
F_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < 0.05)[0]
F_obs_sig = np.nan * np.ones_like(F_obs)
time_inds, space_inds = np.squeeze(clusters[0])

for i, ti in enumerate(time_inds):
    F_obs_sig[ti, space_inds[i]] = 1

ch_inds = np.unique(space_inds)
time_inds = np.unique(time_inds)

plt.imshow(F_obs[:, sen_sorted].T, aspect=15, cmap=parula_map())
plt.imshow(F_obs_sig[:, sen_sorted].T, aspect=15, cmap=parula_map())

t500 = np.argmin(np.abs(erp_times - .5))
ch_mask = np.zeros((n_ch,), dtype=bool)
ch_mask[ch_inds] = True
topoplot_with_colorbar(F_obs[t500], raw_info=raw_info,
                       cmap=parula_map(), mask=ch_mask, vmin=np.min(F_obs), vmax=np.max(F_obs))

# ----------------------------------------------
# FIGURE 5
# ----------------------------------------------

n_source = 8196
thr_source = t.ppf(q=1 - 10 ** (-4) / n_source / 2, df=num_subj - 1)
corr_p300_alpha, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_corr_t')
corr_nop300_alpha, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_corr_s')
t_stat_t_s = np.zeros((n_source,))
for v in range(n_source):
    t_stat_t_s[v] = ttest_rel(corr_p300_alpha[:, v], corr_nop300_alpha[:, v])[0]

data_to_plot = np.multiply(t_stat_t_s, np.abs(t_stat_t_s) > thr_source)
clim = dict(kind='value',
            lims=[-1 * np.nanmax(np.abs(data_to_plot)), 0, 1 * np.nanmax(np.abs(data_to_plot))])
plot_brain_views(data_to_plot, clim, 'corr', cmap=parula_map())
# unhashtag for mask plotting
for i in range(len(data_to_plot)):
    if data_to_plot[i] == 0:
        data_to_plot[i] = -1000
    else:
        data_to_plot[i] = 1000
clim = dict(kind='value',
            lims=[-1 * np.nanmax(np.abs(data_to_plot)), 0, 1 * np.nanmax(np.abs(data_to_plot))])
plot_brain_views(data_to_plot, clim, 'corr_mask', cmap=parula_map())

# ----------------------------------------------------
# FIGURE 6
# ----------------------------------------------------

stc_p300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t', 'pickle')
stc_p300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t_env', 'pickle')
stc_nop300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s', 'pickle')
stc_nop300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s_env', 'pickle')

# ----------------------------------------------
# FIGURE 6a
# ----------------------------------------------

win = np.array([0.3, 0.7])
win_samples = np.array([np.argmin(np.abs(erp_times_dec - win[0])),
                        np.argmin(np.abs(erp_times_dec - win[1]))])
X1 = np.transpose(
    np.mean(stc_p300[:, :, win_samples[0]:win_samples[1]] ** 2, axis=2).reshape((len(ids), n_source, 1)),
    [0, 2, 1])
X2 = np.transpose(
    np.mean(stc_nop300[:, :, win_samples[0]:win_samples[1]] ** 2, axis=2).reshape((len(ids), n_source, 1)),
    [0, 2, 1])

t_vox = [ttest_rel(np.squeeze(X1[:, :, vi]), np.squeeze(X2[:, :, vi]))[0] for vi in range(n_source)]
print(np.sum(np.array(t_vox) > thr_source))

X_avg_diff = np.squeeze(np.mean(X1 - X2, axis=0))
cluster_erp_bool = np.multiply(np.multiply(X_avg_diff, X_avg_diff > np.percentile(X_avg_diff, 90)),
                               np.array(t_vox) > thr_source) > 0

data_to_plot = np.multiply(X_avg_diff, np.array(t_vox) > thr_source)
clim = dict(kind='value', lims=[np.min(X_avg_diff), np.mean(X_avg_diff), np.max(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'p300', cmap=parula_map())

# ----------------------------------------------
# FIGURE 6b
# ----------------------------------------------

X1 = np.transpose(
    np.mean(stc_nop300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2).reshape(
        (len(ids), n_source, 1)), [0, 2, 1]) / \
     np.transpose(
         np.mean(stc_p300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2).reshape(
             (len(ids), n_source, 1)), [0, 2, 1])
X2 = np.ones(X1.shape)

t_vox = [ttest_rel(np.squeeze(X1[:, :, vi]), np.squeeze(X2[:, :, vi]))[0] for vi in range(n_source)]

print(np.sum(np.array(t_vox) > thr_source))

X_avg_diff = np.squeeze(np.mean(X1 / X2, axis=0))
cluster_env_bool = np.multiply(np.multiply(X_avg_diff, X_avg_diff > np.percentile(X_avg_diff, 90)),
                               np.array(t_vox) > thr_source) > 0

data_to_plot = np.multiply(X_avg_diff, np.array(t_vox) > thr_source)
clim = dict(kind='value', lims=[1, np.mean(X_avg_diff), np.max(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'alpha_env', cmap=parula_map_backward())

# plot cluster mask
cluster_bool = np.multiply(cluster_erp_bool, cluster_env_bool)
data_to_plot = cluster_bool
clim = dict(kind='value', lims=[0, np.mean(data_to_plot), np.max(data_to_plot)])
plot_brain_views(data_to_plot, clim, 'cluster')

# ----------------------------------------------
# FIGURE 7a
# ----------------------------------------------

# BSI
stc_bsi, _ = list_from_many(ids, op.join(dir_derr, 'eL_bsi'), '_bsi', 'pickle')
stc_bsi_cluster = np.multiply(stc_bsi, cluster_bool)
stc_bsi_avg = np.mean(stc_bsi, axis=0)

data_to_plot = stc_bsi_avg
clim = dict(kind='value',
            lims=[-1 * np.nanmax(np.abs(data_to_plot)), 0, 1 * np.nanmax(np.abs(data_to_plot))])
plot_brain_views(data_to_plot, clim, 'bsi', cmap=parula_map())

# ----------------------------------------------
# FIGURE 7b
# ----------------------------------------------

bsi_corr = np.array([pearsonr(stc_bsi[:, v], corr_p300_alpha[:, v])[0] for v in range(n_source)])

data_to_plot = bsi_corr
clim = dict(kind='value',
            lims=[np.min(data_to_plot), np.mean(data_to_plot), np.max(data_to_plot)])
plot_brain_views(data_to_plot, clim, 'bsi_corr', cmap=parula_map())

# ----------------------------------------------------
# FIGURE 8
# ----------------------------------------------------
# ----------------------------------------------------
# FIGURE 8b
# ----------------------------------------------------

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

# ----------------------------------------------------
# FIGURE 8c
# ----------------------------------------------------

csp_filter = load_pickle('csp_filter_real', os.getcwd())
csp_pattern = load_pickle('csp_pattern_real', os.getcwd())

topoplot_with_colorbar(csp_pattern[:, 0], raw_info, cmap=parula_map())

# ----------------------------------------------------
# FIGURE 8a
# ----------------------------------------------------

attention_comp, attention_ids = composite_attention(ids)
memory_comp, memory_ids = composite_memory(ids)
executive_comp, executive_ids = composite_executive(ids)

csp_env_peak_lat = load_pickle('csp_env_peak_lat', dir_derr)
csp_env_peak_amp = load_pickle('csp_env_peak_amp_', dir_derr)

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

# --------------------------------------------------
# SUPPLEMENTARY MATERIAL
# --------------------------------------------------

# --------------------------------------------------
# Fig S3a
# --------------------------------------------------

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

# --------------------------------------------------
# Fig S3b
# --------------------------------------------------

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

# -----------------------------------------------
# Fig S4a
# -----------------------------------------------

topoplot_with_colorbar(bsi_mode, raw_info, cmap=parula_map(), vmin=-1, vmax=1)

# -----------------------------------------------
# Fig S4b
# -----------------------------------------------

stc_bsi, _ = list_from_many(ids, op.join(dir_derr, 'eL_bsi'), '_bsi', 'pickle')
bsi_mode = np.zeros((n_source,))
for v in range(n_source):
    n, bins, _ = plt.hist(stc_bsi[:, v], bins=50)
    mode_index = np.argmax(n)
    bsi_mode[v] = (bins[mode_index] + bins[mode_index + 1]) / 2

data_to_plot = bsi_mode
clim = dict(kind='value',
            lims=[-1 * np.nanmax(np.abs(data_to_plot)), 0, 1 * np.nanmax(np.abs(data_to_plot))])
plot_brain_views(data_to_plot, clim, 'bsi_mode', cmap=parula_map())
