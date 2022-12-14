"""
Functions written by other researchers.
"""
import numpy as np
import mne
from scipy.linalg import eig


def _get_global_reject_epochs(raw, decim):
    """
    Author: Denis Engemann <denis.engemann@gmail.com>

    Computes the threshold for global rejection of epochs based on the amplitude
    Parameters
    ________
    raw : instance of mne.Raw
        Raw instance with signals to be cleaned
    decim : int
        decimation factor, used only for computation
    Returns
    -------
    reject : dict
        threshold for various types of channels, key - sensor type, value - rejection threshold
    """

    from autoreject import get_rejection_threshold

    duration = 3.
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0, duration=duration)

    epochs = mne.Epochs(
        raw, events, event_id=3000, tmin=0, tmax=duration, proj=False,
        baseline=None, reject=None)
    epochs.apply_proj()
    epochs.load_data()
    epochs.pick_types(eeg=True)
    reject = get_rejection_threshold(epochs, decim=decim)
    return reject


def compute_ged(cov_signal, cov_noise):
    """
    The code is from https://github.com/nschawor/eeg-leadfield-mixing/blob/main/code/ssd.py
    by Schaworonkow Natalie

    Functions to compute Spatial-Spectral Decompostion (SSD).
    Reference
    ---------
    Nikulin VV, Nolte G, Curio G.: A novel method for reliable and fast
    extraction of neuronal EEG/MEG oscillations on the basis of
    spatio-spectral decomposition. Neuroimage. 2011 Apr 15;55(4):1528-35.
    doi: 10.1016/j.neuroimage.2011.01.057. Epub 2011 Jan 27. PMID: 21276858.

    Compute a generatlized eigenvalue decomposition maximizing principal
    directions spanned by the signal contribution while minimizing directions
    spanned by the noise contribution.
    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    cov_noise : array, 2-D
        Covariance matrix of the noise contribution.
    Returns
    -------
    filters : array
        SSD spatial filter matrix, columns are individual filters.
    """

    nr_channels = cov_signal.shape[0]

    # check for rank-deficiency
    [lambda_val, filters] = eig(cov_signal)
    idx = np.argsort(lambda_val)[::-1]
    filters = np.real(filters[:, idx])
    lambda_val = np.real(lambda_val[idx])
    tol = lambda_val[0] * 1e-6
    r = np.sum(lambda_val > tol)

    # if rank smaller than nr_channels make expansion
    if r < nr_channels:
        print("Warning: Input data is not full rank")
        M = np.matmul(filters[:, :r], np.diag(lambda_val[:r] ** -0.5))
    else:
        M = np.diag(np.ones((nr_channels,)))

    cov_signal_ex = (M.T @ cov_signal) @ M
    cov_noise_ex = (M.T @ cov_noise) @ M

    [lambda_val, filters] = eig(cov_signal_ex, cov_signal_ex + cov_noise_ex)

    # eigenvalues should be sorted by size already, but double checking
    idx = np.argsort(lambda_val)[::-1]
    filters = filters[:, idx]
    filters = np.matmul(M, filters)

    return filters


def compute_patterns(cov_signal, filters):
    """
    The code is from https://github.com/nschawor/eeg-leadfield-mixing/blob/main/code/ssd.py
    by Schaworonkow Natalie

    Functions to compute Spatial-Spectral Decompostion (SSD).
    Reference
    ---------
    Nikulin VV, Nolte G, Curio G.: A novel method for reliable and fast
    extraction of neuronal EEG/MEG oscillations on the basis of
    spatio-spectral decomposition. Neuroimage. 2011 Apr 15;55(4):1528-35.
    doi: 10.1016/j.neuroimage.2011.01.057. Epub 2011 Jan 27. PMID: 21276858.

    Compute spatial patterns for a specific covariance matrix.
    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    Returns
    -------
    patterns : array, 2-D
        Spatial patterns.
    """

    top = cov_signal @ filters
    bottom = (filters.T @ cov_signal) @ filters
    patterns = top @ np.linalg.pinv(bottom)

    return patterns
