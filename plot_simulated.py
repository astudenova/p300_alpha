"""
This code plots the figures in Results.
"""
import os
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tools_general import list_from_many, load_json_to_numpy, load_pickle, load_json
from tools_plotting import plot_with_sem_one_line, topoplot_with_colorbar, parula_map, \
    parula_map_backward, plot_brain_views
from tools_signal import pk_latencies_amplitudes, lda_

mpl.use("Qt5Agg")

dir_codes = os.getcwd()
dir_derr = load_json('settings/dirs_files.json', os.getcwd())['dir_save']
erp_times = np.array(load_json('erp_times.json', dir_codes))
raw_info = load_pickle('raw_info.pkl', dir_codes)
ids = load_json('settings/ids.json', dir_codes)
full_mask = load_pickle('settings/full_mask.pkl', dir_codes)
num_subj = len(ids)
n_ch = 31

# for source reconstruction
subjects_dir = load_json('settings/dirs_files.json', os.getcwd())['subjects_dir']
subject = 'fsaverage'

erp_times_dec = load_json_to_numpy('erp_times_dec.json', dir_codes)
stc_fixed = load_pickle('stc_fixed.pkl', dir_codes)

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
                       ylim=[-1 * 10 ** (-3), 3.5 * 10 ** (-3)],
                       color_y='darkblue', color_y_sem='skyblue', label_y='target')
plot_with_sem_one_line(erp_times, avg_erp_s[:, pz_idx, :], ax, 0, xlim=[-0.2, 1.1],
                       ylim=[-1 * 10 ** (-3), 3.5 * 10 ** (-3)],
                       color_y='steelblue', color_y_sem='powderblue', label_y='standard')
plot_with_sem_one_line(erp_times, avg_env_t[:, pz_idx, :], ax, 1, xlim=[-0.2, 1.1],
                       ylim=[1 * 10 ** (-3), 4.5 * 10 ** (-3)],
                       color_y='orange', color_y_sem='moccasin', label_y='target')
plot_with_sem_one_line(erp_times, avg_env_s[:, pz_idx, :], ax, 1, xlim=[-0.2, 1.1],
                       ylim=[1 * 10 ** (-3), 4.5 * 10 ** (-3)],
                       color_y='tan', color_y_sem='antiquewhite', label_y='standard')

# ---------------------------------------------------------------
# FIGURE 4
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

# ---------------------------------------------------------------
# FIGURE 5a
# ---------------------------------------------------------------

# This file is generated with the script p_compute_bsi.py
bsi_all = load_pickle('bsi_all', dir_derr)[full_mask]
bsi_mean = np.mean(bsi_all, axis=0)
topoplot_with_colorbar(bsi_mean, raw_info, cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 6
# ---------------------------------------------------------------

# These files are generated with the script p_source_reconstruction.py
stc_p300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t', 'pickle')
stc_p300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_t_env', 'pickle')
stc_nop300, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s', 'pickle')
stc_nop300_alpha_env, _ = list_from_many(ids, op.join(dir_derr, 'eL_p300_alpha'), '_s_env', 'pickle')
n_source = 8196

# ---------------------------------------------------------------
# FIGURE 6a
# ---------------------------------------------------------------

win = np.array([0.3, 0.7])
win_samples = np.array([np.argmin(np.abs(erp_times_dec - win[0])),
                        np.argmin(np.abs(erp_times_dec - win[1]))])
X1 = np.transpose(
    np.mean(stc_p300[:, :, win_samples[0]:win_samples[1]] ** 2, axis=2).reshape((len(ids), n_source, 1)),
    [0, 2, 1])
X2 = np.transpose(
    np.mean(stc_nop300[:, :, win_samples[0]:win_samples[1]] ** 2, axis=2).reshape((len(ids), n_source, 1)),
    [0, 2, 1])

X_avg_diff = np.squeeze(np.mean(X1 - X2, axis=0))

data_to_plot = X_avg_diff
clim = dict(kind='value', lims=[np.min(X_avg_diff), np.mean(X_avg_diff), np.max(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'p300', cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 6b
# ---------------------------------------------------------------

X1 = np.transpose(
    np.mean(stc_nop300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2).reshape(
        (len(ids), n_source, 1)), [0, 2, 1]) / \
     np.transpose(
         np.mean(stc_p300_alpha_env[:, :, win_samples[0]:win_samples[1]], axis=2).reshape(
             (len(ids), n_source, 1)), [0, 2, 1])
X2 = np.ones(X1.shape)

X_avg_diff = np.squeeze(np.mean(X1 / X2, axis=0))

data_to_plot = X_avg_diff
clim = dict(kind='value', lims=[1, np.mean(X_avg_diff), np.max(X_avg_diff)])
plot_brain_views(data_to_plot, clim, 'alpha_env', cmap=parula_map_backward())

# ---------------------------------------------------------------
# FIGURE 7b
# ---------------------------------------------------------------

lda_filter, lda_pattern = lda_(avg_erp_t[:, :n_ch], avg_erp_s[:, :n_ch], [0.3, 0.7], erp_times)
topoplot_with_colorbar(lda_pattern, raw_info, cmap=parula_map())

# ---------------------------------------------------------------
# FIGURE 7c
# ---------------------------------------------------------------

# This file is generated with the script p_save_covariance_apply_csp.py
csp_pattern = load_pickle('csp_pattern', os.getcwd())

topoplot_with_colorbar(csp_pattern[:, 0], raw_info, cmap=parula_map())
