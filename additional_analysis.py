"""
This code contains additional analysis suggested by reviewers.
"""
import os
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test, ttest_ind_no_p
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr, ttest_rel, t, spearmanr, zscore
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

full_mask = np.logical_not(np.in1d(np.arange(0, len(ids_all)), id_mask))
# These files are generated with the script p_read_erp_alpha_save.py
avg_erp_t = load_pickle('avg_erp_t', dir_derr)[full_mask]
avg_erp_s = load_pickle('avg_erp_s', dir_derr)[full_mask]
avg_erp_t_nobl = load_pickle('avg_erp_t_nobl', dir_derr)[full_mask]  # not baselined
avg_env_t = load_pickle('avg_env_t', dir_derr)[full_mask]
avg_env_s = load_pickle('avg_env_s', dir_derr)[full_mask]

pz_idx = np.where(np.array(raw_info.ch_names) == 'Pz')[0][0]

# ---------------------------------------------------------------
# correlations with total intracranial volume
# ---------------------------------------------------------------

# read TIV
tiv = pd.read_csv('ids_w_estimatedTIV.csv')
tiv_data = tiv['EstimatedTotalIntraCranialVol'].to_numpy()
tiv_mask = ~np.isnan(tiv_data)

# compute peak amplitudes and latencies
erp_peaks_avg = []
for i_subj, subj in enumerate(ids):
    avg_erp_pk = pk_latencies_amplitudes(avg_erp_t[i_subj, pz_idx].reshape((1, -1)),
                                         np.array([0.2, 1]), erp_times, direction='pos')[0][1]
    if avg_erp_pk != 0:
        erp_peaks_avg.append(avg_erp_pk)
    else:
        erp_peaks_avg.append(.5)
        print('ERP peak is not found for subject ' + str(subj))
        print('Setting the peak latency to 0.5.')

# compute erd at P300 peak
erd_avg = np.zeros((len(ids), n_ch))
for i_subj in range(len(ids)):
    pk_sample_erp = np.argmin(np.abs(erp_times - erp_peaks_avg[i_subj]))
    erd_avg[i_subj] = (np.mean(avg_env_t[i_subj, :, pk_sample_erp - 50:pk_sample_erp + 50], axis=1) / np.mean(
        avg_env_t[i_subj, :, 150:250], axis=1) - 1)

# compute P300 amplitude and alpha envelope amplitude around P300 peak
erp_topo_avg = np.zeros((len(ids), n_ch))
env_topo_avg = np.zeros((len(ids), n_ch))
for i_subj in range(len(ids)):
    pk_sample_erp = np.argmin(np.abs(erp_times - erp_peaks_avg[i_subj]))
    env_topo_avg[i_subj] = np.mean(avg_env_t[i_subj, :, pk_sample_erp - 50:pk_sample_erp + 50], axis=1)
    erp_topo_avg[i_subj] = avg_erp_t[i_subj, :n_ch, pk_sample_erp]

# read BSI
bsi_all = load_pickle('bsi_all', dir_derr)[full_mask]

print(str(np.sum(tiv_mask)) + ' participants had the TIV data.')

tiv_corr = np.zeros((n_ch, 4, 2))  # with P300 peak amplitude, with alpha trough amplitude, with erd, with bsi
for ch in range(n_ch):
    tiv_corr[ch, 0, 0] = pearsonr(tiv_data[tiv_mask], erp_topo_avg[tiv_mask, ch])[0]
    tiv_corr[ch, 0, 1] = pearsonr(tiv_data[tiv_mask], erp_topo_avg[tiv_mask, ch])[1]

    tiv_corr[ch, 1, 0] = pearsonr(tiv_data[tiv_mask], env_topo_avg[tiv_mask, ch])[0]
    tiv_corr[ch, 1, 1] = pearsonr(tiv_data[tiv_mask], env_topo_avg[tiv_mask, ch])[1]

    tiv_corr[ch, 2, 0] = pearsonr(tiv_data[tiv_mask], erd_avg[tiv_mask, ch])[0]
    tiv_corr[ch, 2, 1] = pearsonr(tiv_data[tiv_mask], erd_avg[tiv_mask, ch])[1]

    tiv_corr[ch, 3, 0] = pearsonr(tiv_data[tiv_mask], np.abs(bsi_all[tiv_mask, ch]))[0]
    tiv_corr[ch, 3, 1] = pearsonr(tiv_data[tiv_mask], np.abs(bsi_all[tiv_mask, ch]))[1]

for iplot in range(4):
    topoplot_with_colorbar(tiv_corr[:, iplot, 0], raw_info=raw_info,
                           cmap=parula_map(), mask=tiv_corr[:, iplot, 1] < 0.05 / n_ch)

# ---------------------------------------------------------------
# Fig R1
# ---------------------------------------------------------------
# re-baseline to peak latency
avg_erp_t_300bl = np.zeros(avg_erp_t_nobl.shape)
avg_env_t_300bl = np.zeros(avg_env_t.shape)
for i_subj in range(len(ids)):
    bl_win = [np.argmin(np.abs(erp_times - .4)), np.argmin(np.abs(erp_times - .6))]
    bl = np.mean(avg_erp_t_nobl[i_subj][:, bl_win[0]:bl_win[1]], axis=1)
    avg_erp_t_300bl[i_subj] = np.subtract(avg_erp_t_nobl[i_subj], bl.reshape((-1, 1)))
    bl = -np.mean(avg_env_t[i_subj][:, 150:250], axis=1)
    avg_env_t_300bl[i_subj] = np.subtract(-avg_env_t[i_subj], bl.reshape((-1, 1)))

erd_pz = erd_avg[:, pz_idx]

# create bins
erd_bins = list(np.percentile(erd_pz, np.arange(0, 100, 20)))
erd_bins.append(np.max(erd_pz))
erd_bins = np.array(erd_bins)
binned_idx = np.zeros(len(erd_pz), )
for ai in range(5):
    binned_idx = binned_idx + ai * ((erd_bins[ai] <= erd_pz) * (erd_pz < erd_bins[ai + 1]))

# plot
fig, ax = plt.subplots(1, 3)
ax = ax.flatten()
color_bins_erp = ['#00406B', '#104E8B', '#1874CD', '#1C86EE', '#1EA1FF']
color_bins_env = ['#3B1E00', '#633200', '#8B4500', '#B35900', '#E57600']
label_erd = ['ERD-bin1: ' + str("{:.2f}".format(erd_bins[0])) + ' - ' + str("{:.2f}".format(erd_bins[1])),
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
                    label_y=label_erd)
ax[0].set_title('ERP, baseline before 0')
ax[0].set_xlim([-0.2, 1.1])
plot_with_sem_color(erp_times, [avg_erp_t_nobl[binned_idx == 0, pz_idx, :], avg_erp_t_nobl[binned_idx == 1, pz_idx, :],
                                avg_erp_t_nobl[binned_idx == 2, pz_idx, :], avg_erp_t_nobl[binned_idx == 3, pz_idx, :],
                                avg_erp_t_nobl[binned_idx == 4, pz_idx, :]],
                    ax, 1, xlim=[-.3, 1.25],
                    color_y=color_bins_erp,
                    color_y_sem=color_bins_erp,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=len(label_erd) * [None])
ax[1].set_title('ERP no baseline')
ax[1].set_xlim([-0.2, 1.1])
plot_with_sem_color(erp_times,
                    [avg_erp_t_300bl[binned_idx == 0, pz_idx, :], avg_erp_t_300bl[binned_idx == 1, pz_idx, :],
                     avg_erp_t_300bl[binned_idx == 2, pz_idx, :], avg_erp_t_300bl[binned_idx == 3, pz_idx, :],
                     avg_erp_t_300bl[binned_idx == 4, pz_idx, :]],
                    ax, 2, xlim=[-.3, 1.25],
                    color_y=color_bins_erp,
                    color_y_sem=color_bins_erp,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=len(label_erd) * [None])
ax[2].set_title('ERP baseline in P300 window')
ax[2].set_xlim([-0.2, 1.1])
fig, ax = plt.subplots(1, 2)
plot_with_sem_color(erp_times, [avg_env_t[binned_idx == 0, pz_idx, :], avg_env_t[binned_idx == 1, pz_idx, :],
                                avg_env_t[binned_idx == 2, pz_idx, :], avg_env_t[binned_idx == 3, pz_idx, :],
                                avg_env_t[binned_idx == 4, pz_idx, :]],
                    ax, 0, xlim=[-.3, 1.25],
                    color_y=color_bins_env,
                    color_y_sem=color_bins_env,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=label_erd)
ax[0].set_title('alpha envelope')
ax[0].set_xlim([-0.2, 1.1])
plot_with_sem_color(erp_times,
                    [avg_env_t_300bl[binned_idx == 0, pz_idx, :], avg_env_t_300bl[binned_idx == 1, pz_idx, :],
                     avg_env_t_300bl[binned_idx == 2, pz_idx, :], avg_env_t_300bl[binned_idx == 3, pz_idx, :],
                     avg_env_t_300bl[binned_idx == 4, pz_idx, :]],
                    ax, 1, xlim=[-.3, 1.25],
                    color_y=color_bins_env,
                    color_y_sem=color_bins_env,
                    alpha_level=[0.1, 0.1, 0.1, 0.1, 0.1],
                    label_y=len(label_erd) * [None])
ax[1].set_title('alpha envelope multiplied by -1')
ax[1].set_xlim([-0.2, 1.1])

# statistical evaluation
erp_to_test = avg_erp_t_300bl

# make bins
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
    ampl_full_t[0, i_subj] = np.multiply(erp_to_test[i_subj, :n_ch].T, subj_mask_bin1).T
    ampl_full_t[1, i_subj] = np.multiply(erp_to_test[i_subj, :n_ch].T, subj_mask_bin5).T

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

t100 = np.argmin(np.abs(erp_times + 0.1))  # pre-stimulus window
ch_mask = np.zeros((n_ch,), dtype=bool)
ch_mask[F_obs_sig[t100] != 1] = True
topoplot_with_colorbar(F_obs[t100], raw_info=raw_info,
                       cmap=parula_map(), mask=ch_mask, vmin=np.min(F_obs), vmax=np.max(F_obs))
