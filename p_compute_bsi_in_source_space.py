"""
This pipeline computes baseline-shift indices in source space.
"""
import os
import os.path as op
import mne
import numpy as np
from tools_general import load_json, save_pickle, load_pickle, specify_dirs
from tools_lifedataset import read_rest, create_raw_for_source_reconstruction
from tools_signal import bsi, create_noise_cov

dirs = specify_dirs()
dir_save = dirs['dir_save']
ids = load_json('ids', os.getcwd())
alpha_peaks = load_pickle('alpha_peaks', os.getcwd())
markers_rest = load_json('markers_rest', os.getcwd())

# folders for source reconstruction
subjects_dir = dirs['subjects_dir']
subject = 'fsaverage'
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct6' + '-fwd.fif')

for i_subj, subj in enumerate(ids):

    # read resting-state recording
    raw = read_rest(subj, markers_rest[subj])

    fs = raw.info['sfreq']
    raw_data = raw.get_data(picks='eeg')
    n_times = raw_data.shape[1]
    # prepare raw for source reconstruction
    evoked_raw = create_raw_for_source_reconstruction(raw)
    # create noise covariance with a bias of data length
    noise_cov = create_noise_cov(evoked_raw.data.shape, evoked_raw.info)

    # source reconstruction with eLORETA
    forward = mne.read_forward_solution(fwd_dir)
    inv_op = mne.minimum_norm.make_inverse_operator(evoked_raw.info, forward, noise_cov,
                                                    loose=1.0, depth=5, fixed=False)
    stc_el = mne.minimum_norm.apply_inverse(evoked_raw.copy(), inverse_operator=inv_op,
                                            lambda2=0.05, method='eLORETA', pick_ori='normal')

    # compute BSI for each voxel
    stc_el_data = stc_el._data
    n_vox = stc_el_data.shape[0]
    bsi_el = np.zeros((n_vox,))
    alpha_pk = np.mean(alpha_peaks[i_subj])
    for vi in range(n_vox):
        bsi_el[vi] = bsi(stc_el_data[vi], fs=stc_el.sfreq, alpha_peak=alpha_pk)
    save_pickle(subj + '_bsi', op.join(dir_save, 'eL_bsi'), bsi_el)

    print('--------------' + str(i_subj) + ' ' + subj + ' is finished--------------------')
