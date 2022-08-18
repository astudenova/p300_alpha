"""
Functions for signal processing and data processing.
"""
import numpy as np
import mne
from scipy.signal import butter, filtfilt, welch, find_peaks


def peak_in_spectrum(Pxx, freq):
    """
    Computes peak frequency in the alpha band 8-12 Hz from the given spectrum

    :param numpy.ndarray Pxx: amplitude spectrum, recommended resolution - 0.1 Hz,
        recommended window - more than 5 sec (for smoothness) (dimensions - 1D)
    :param numpy.ndarray freq: frequencies corresponding to the spectrum (dimensions - 1D)
    :returns: (float) - peak frequency in the alpha band
    """

    # find peak that are sufficiently far apart
    peak_locs, _ = find_peaks(Pxx, distance=80)

    # find all peaks that landed into the 8-12 range
    alphapks = [freq[pk] for pk in peak_locs if 8 < freq[pk] < 12]

    # if multiple peaks were found, take average
    # if no peaks were found, set peak frequency to 10 Hz

    return float(np.mean(alphapks)) if alphapks else 10.0


def filtfilt_with_padding(signal, b_coeff, a_coeff, padlen):
    """
    Filters signal using manual padding with mirroring

    :param numpy.ndarray signal: signal to filter (dimensions - 1D)
    :param numpy.ndarray b_coeff: filter coefficients (dimensions - 1D)
    :param numpy.ndarray a_coeff: filter coefficients (dimensions - 1D)
    :param int padlen: the desirable window for padding
    :returns: (numpy.ndarray, 1D) - filtered signal
    """

    # make sure it is a column vector
    if signal.shape[0] == 1:
        signal = signal.reshape((-1))

    x_pad = np.hstack((np.flip(signal)[-padlen:], signal, np.flip(signal[-padlen:])))
    x_padded = filtfilt(b_coeff, a_coeff, x_pad, padtype=None)

    return x_padded[padlen:-padlen]


def bsi_pearson(x_ampl, x_lowpass, n_bins=20):
    """
    Computes the baseline-shift index
    Computation is based on Nikulin et al., Clinical Neurophysiology, 2010

    :param numpy.ndarray x_ampl: amplitude envelope of frequency band of interest
        (dimensions - 1D)
    :param numpy.ndarray x_lowpass: filtered signal in the band that roughly corresponds
        to x_ampl frequency content (dimensions - 1D)
    :param int n_bins: number of bins for binning of the amplitude, default = 20
    :returns:
        bsi (float) - the baseline-shift index for a given set of two signals,
            which is essentially just a degree of correlation
        V_alpha (numpy.ndarray, 1D) - array of average values in each bin
            for the frequency band of interest
        V_bs (numpy.ndarray, 1D) - array of average values in each bin for the baseline shifts
    """

    binned_ampl = np.zeros(len(x_ampl), )
    V_alpha = np.zeros((n_bins,))
    V_bs = np.zeros((n_bins,))
    # bin data
    bin_step = int(100 / n_bins)
    bin_borders = np.percentile(np.sort(x_ampl), np.arange(0, 100, bin_step))
    bin_borders = np.append(bin_borders, np.max(x_ampl))
    for ai in range(n_bins):
        binned_ampl = binned_ampl + ai * ((bin_borders[ai] <= x_ampl) * (x_ampl < bin_borders[ai + 1]))

    # fill arrays with mean values of the bins
    for bi in range(n_bins):
        V_alpha[bi] = np.mean(x_ampl[binned_ampl == bi])
        V_bs[bi] = np.mean(x_lowpass[binned_ampl == bi])

    # calculate the slope
    M_l = np.vstack((np.ones((n_bins,)), V_alpha)).T
    M_r = V_bs
    c, _, _, _ = np.linalg.lstsq(M_l, M_r, rcond=None)
    # turn the slope into Pearson correlation coefficient
    bsi = c[1] * (np.std(V_alpha) / np.std(V_bs))

    return bsi, V_alpha, V_bs


def bsi(signal, fs, padlen=50000, alpha_peak=None):
    """
    Precomputes signals for further the baseline-shift index calculation with bsi_pearson

    :param numpy.ndarray X: signal for which computation will be carries out (dimensions - 1D)
    :param float fs: sampling frequency
    :param int padlen: number of samples that will be used for padding the signal
        when filtering, default = 50000 samples
    :param alpha_peak: peak frequency of the band of interest if precomputed before, default = None
    :returns: bsi (float) - the baseline-shift index
    """

    from scipy.signal import hilbert

    # if alpha_peak is not precomputed, compute it here
    if alpha_peak is None:
        f, Pxx = welch(signal, fs=fs, nperseg=int(10 * fs), noverlap=int(5 * fs),
                       nfft=int(10 * fs), detrend=False)
        alpha_peak = peak_in_spectrum(Pxx, f)

    x_lowpass = filter_in_low_frequency(signal, fs, padlen=padlen, padding_mirror=True)
    x_passband = filter_in_alpha_band(signal, fs, padlen=padlen, alpha_peak=alpha_peak, padding_mirror=True)
    x_ampl = np.abs(hilbert(x_passband))

    bsi, _, _ = bsi_pearson(x_ampl, x_lowpass)

    return bsi


def filter_in_low_frequency(signal, fs, padlen, padding_mirror=False):
    """
    Filters input in low-frequency band at 0-3Hz

    :param numpy.ndarray signal: signal to filter (dimensions - 1D if padding_mirror=True,
        otherwise can be 2D, 3D)
    :param float fs: sampling frequency
    :param padlen: the length of padded samples
    :param bool padding_mirror: whether to use mirroring when padding,
        default=False - using zero padding
    :return: (numpy.ndarray) - filtered signal
    """

    lf_cutoff = 3
    b_lf, a_lf = butter(N=4, Wn=lf_cutoff / fs * 2, btype='lowpass')
    if padding_mirror:
        return filtfilt_with_padding(signal, b_lf, a_lf, padlen)
    else:
        return filtfilt(b_lf, a_lf, signal, padlen=padlen)


def compute_envelope(data, n_epoch, n_ch, fs, alpha_peaks=None, subtract=False):
    """
    Computes amplitude envelope from epoched data

    :param numpy.ndarray epochs: broadband data (dimensions - 3D)
    :param int n_epoch: number of epochs
    :param int n_ch: number of channels in the data
    :param numpy.ndarray alpha_peaks: peak frequencies for each channel, if None
        will be computed inside the function (dimensions - 1D)
    :param bool subtract: flag to whether on not subtract average ER from each trial
    :returns: env_epoch (numpy.ndarray, 3D) - amplitude envelope
        with dimensions epochs x channels x time
    """
    from harmoni.extratools import hilbert_

    # if array is not 3D - return an error
    if len(data.shape) != 3:
        raise ValueError('Input data should be 3D.')

    data, epochs, channels, times = check_dimensions_3D(data, n_epoch, n_ch)

    env_epoch = np.zeros((data.shape))

    # compute spectrum for each channel
    if alpha_peaks is None:
        data_cont = np.array([data[:, i, :].reshape((-1)) for i in range(channels)])
        f, Pxx = welch(data_cont, fs=fs, nperseg=10 * fs, noverlap=5 * fs,
                       nfft=10 * fs, detrend=False)
        alpha_peaks = np.array([peak_in_spectrum(Pxx[i_ch], f) for i_ch in range(channels)])
    elif len(alpha_peaks) != channels:
            raise ValueError('alpha_peaks should contain value for each channel.')

    # compute average ER
    if subtract:
        epochs_avg = np.mean(data, axis=0)
        data = np.array([data[i_ep] - epochs_avg for i_ep in range(epochs)])

    # estimate padlen
    padlen = int(times / 5) if times < 2000 else 50000

    for i_ch in range(channels):
        alpha_peak = alpha_peaks[i_ch]
        alpha = filter_in_alpha_band(data[:, i_ch], fs, padlen=padlen, alpha_peak=alpha_peak)
        # extract envelope
        env_epoch[:, i_ch] = np.abs(hilbert_(alpha))

    return env_epoch


def filter_in_alpha_band(signal, fs, padlen, alpha_peak=10, padding_mirror=False):
    """
    Filters input in alpha band based on peak frequency +/- 2Hz

    :param numpy.ndarray signal: signal to filter (dimensions - 1D if padding_mirror=True,
        otherwise can be 2D, 3D)
    :param float fs: sampling frequency
    :param padlen: the length of padded samples
    :param alpha_peak: peak frequency, default=10 Hz
    :param bool padding_mirror: whether to use mirroring when padding,
        default=False - using zero padding
    :return: (numpy.ndarray) - filtered signal
    """

    adj_band = np.array([alpha_peak - 2, alpha_peak + 2])
    b10, a10 = butter(N=2, Wn=adj_band / fs * 2, btype='bandpass')
    # filter in the alpha band
    if padding_mirror:
        return filtfilt_with_padding(signal, b10, a10, padlen)
    else:
        return filtfilt(b10, a10, signal, padlen=padlen, axis=1)


def project_eog(raw, decim=1, reject_with_threshold=False):
    """
    Computes the projections for eye-movements subspace, add projections to raw

    Adapted from Denis Engemann <denis.engemann@gmail.com>

    :param mne.Raw raw: Raw instance to which projections will be added
    :param int decim: decimation factor, used only for computation, default = 1 (no decimation)
    :param bool reject_with_threshold: additional rejection
        of eye-movement epochs based on the amplitude, default False
    """
    from autoreject import get_rejection_threshold

    reject_eog = None
    if reject_with_threshold:
        raw_copy = raw.copy()
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_copy, ch_name=['VEOG', 'HEOG'])
        if len(eog_epochs) >= 5:
            reject_eog = get_rejection_threshold(eog_epochs, decim=decim)
            del reject_eog['eog']

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1, ch_name=['VEOG', 'HEOG'])

    raw.add_proj(proj_eog[0])


def pk_latencies_amplitudes(data, win, times, direction):
    """
    Computes peak latency and amplitude of a slow-varying process such as
    evoked response or amplitude change

    :param numpy.ndarray data: signal for which computation will be carried out
        (dimensions - 2D, epochs x time)
    :param numpy.ndarray win: window in time, where peak should be present,
        2 values - start and stop (dimensions - 1D)
    :param numpy.ndarray times: vector of time, corresponding to the data signal (dimensions - 1D)
    :param str direction: 'pos' or 'neg' direction of the peak to be found, positive is upwards, negative is downwards
    :returns: peaks (numpy.ndarray, 2D) - array with peak sample, peak latency,
        and peak amplitude computed for each epoch, i.e. with a shape (n_epochs x 3)
    """

    # if array is not 2D - return an error
    if len(data.shape) > 2:
        raise ValueError('Input data should be 2D.')
    # if data is a vector, add one dimension
    if len(data.shape) < 2:
        data = data[np.newaxis, :]
    # reshape the data, if first dimension is time
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_epoch = data.shape[0]
    win_samples = np.array([np.argmin(np.abs(times - win[0])), np.argmin(np.abs(times - win[1]))])

    peaks = np.zeros((n_epoch, 3))
    for e_i in range(n_epoch):
        # cut the chunk of data according to predefined window
        epoch = data[e_i, win_samples[0]:win_samples[1]]

        if direction == 'pos':
            peak_locs, _ = find_peaks(epoch)
        elif direction == 'neg':  # if direction is negative flip the signal
            peak_locs, _ = find_peaks(-epoch)
        else:
            raise ValueError('The direction should be either \'pos\' or \'neg\'.')

        # if peaks are not found, quit the computations for current epoch
        if np.sum(peak_locs) == 0:
            print('Peaks are not found for epoch #' + str(e_i))
            continue
        # account for the window
        peak_locs = peak_locs + win_samples[0]
        pk_time = times[peak_locs]
        pk_ampl = epoch[peak_locs - win_samples[0]]
        # if several peaks, take the most extreme one
        if len(pk_ampl) > 1:
            idx_max = np.argmax(pk_ampl) if direction == 'pos' else np.argmin(pk_ampl)
            peaks[e_i] = [peak_locs[idx_max], pk_time[idx_max], pk_ampl[idx_max]]
        else:
            peaks[e_i] = [peak_locs, pk_time, pk_ampl]

    return peaks


def create_noise_cov(data_size, data_info):
    """
    Computes identity noise covariance with a bias of data length

    Method is by Mina Jamshidi Idaji (minajamshidi91@gmail.com)

    :param tuple data_size: size of original data (dimensions - 1D)
    :param mne.Info data_info: info that corresponds to the original data
    :returns: (mne.Covariance) - noise covariance for further source reconstruction
    """

    data1 = np.random.normal(loc=0.0, scale=1.0, size=data_size)
    raw1 = mne.io.RawArray(data1, data_info)

    return mne.compute_raw_covariance(raw1, tmin=0, tmax=None)


def lda_(signal, noise, win, time_vec):
    """
    Computes LDA weights for two signals

    The method is based on Blankertz et al., Single-trial analysis and classification
    of ERP components - a tutorial, NeuroImage, 2011

    The code is adapted from https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

    :param numpy.ndarray signal: the data that will be treated as signal,
        epochs x channels x time (dimensions - 3D)
    :param numpy.ndarray noise: the data that will be treated as noise,
        epochs x channels x time (dimensions - 3D)
    :param list win: time window in which signals will be averaged
    :param numpy.ndarray time_vec: time that corresponds to the data (dimensions - 1D)
    :returns:
        clf_weights (numpy.array, 1D) - weights of the LDA filter, have dimensions (channels,)
        clf_pattern(numpy.array, 1D) - spatial pattern that corresponds to LDA filter,
        have dimensions (channels,)
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # if array is not 3D - return an error
    if len(signal.shape) != 3 or len(noise.shape) != 3:
        raise ValueError('Input data should be 3D.')

    # compute samples corresponding to window
    win_samples = np.array([np.argmin(np.abs(time_vec - win[0])),
                            np.argmin(np.abs(time_vec - win[1]))])
    # average signal and noise within predefined window
    sig_avg = np.mean(signal[:, :, win_samples[0]:win_samples[1]], axis=2)
    noise_avg = np.mean(noise[:, :, win_samples[0]:win_samples[1]], axis=2)
    # prepare arrays for LDA
    trials = np.hstack((np.ones((signal.shape[0],)), np.zeros((signal.shape[0],))))
    data = np.vstack((sig_avg,
                      noise_avg[np.random.random_integers(0, noise.shape[0] - 1, size=signal.shape[0])]))
    # run lda
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(data, trials)
    clf_weights = clf.coef_.T
    cov_data = np.cov(data.T)
    clf_pattern = (cov_data @ clf_weights) @ np.linalg.pinv((clf_weights.T @ cov_data) @ clf_weights)

    return clf_weights.reshape((-1)), clf_pattern.reshape((-1))


def from_cont_to_epoch(data, n_epoch, n_times):
    """
    Reshapes continuous data with dimensions channels x time
    to epoched data with dimensions epochs x channels x time

    :param numpy.ndarray data: data to reshape (dimensions - 2D)
    :param int n_epoch: number of epochs
    :param int n_times: number of time points in each epoch
    :returns: data_ep (numpy.ndarray, 3D) - epoched data with dimensions epochs x channels x time
    """
    # if data is not 2D - return an error
    if len(data.shape) != 2:
        raise ValueError('Input data should be 2D.')

    (_, times) = data.shape
    # if data array is time x channels - reshape
    if n_epoch * n_times != times:
        data = data.T

    data_ep = np.array([data[:, e_i * n_times:(e_i + 1) * n_times] for e_i in range(n_epoch)])

    return data_ep


def from_epoch_to_cont(data, n_ch, n_epoch, wings=None):
    """
    Reshapes epoched data with dimensions epochs x channels x time
    to continuous data with dimensions channels x time

    :param numpy.ndarray data: data to reshape, epochs x channels x time (dimensions - 3D)
    :param int n_ch: number of channels in the data
    :param int n_epoch: number of epochs
    :param int wings: number of sample to cut from each side to avoid artifacts, default = None
    :returns: data_cont (numpy.ndarray, 3D) - continuous data with dimensions channels x time
    """
    # if array is not 3D - return an error
    if len(data.shape) != 3:
        raise ValueError('Input data should be 3D.')

    data, epochs, channels, times = check_dimensions_3D(data, n_epoch, n_ch)

    if wings is None:
        data_cont = np.zeros((channels, epochs * times))
        for i in range(epochs):
            data_cont[:, i * times:(i + 1) * times] = data[i, :channels]
    else:
        data_cont = np.zeros((channels, epochs * (times - 2 * wings)))
        for i in range(epochs):
            data_cont[:, i * (times - 2 * wings):(i + 1) * (times - 2 * wings)] = \
                data[i, :channels, wings:-wings]

    return data_cont


def apply_spatial_filter(data, spat_filter, spat_pattern, n_ch, n_epoch):
    """
    Applies spatial filter to the data and normalized on spatial pattern

    :param numpy.ndarray data: data to filter, epochs x channels x time (dimensions - 3D)
    :param numpy.ndarray spat_filter: a single spatial filter, the length of the vector
        should be equal number of channels (dimensions - 1D)
    :param numpy.ndarray spat_pattern: a corresponding spatial pattern, the length
        of the vector should be equal number of channels (dimensions - 1D)
    :param int n_ch: number of channels in the data
    :param int n_epoch: number of epochs
    :returns: data_spat (numpy.ndarray, 2D) - continuous data with dimensions epochs x time
    """
    # if array is not 3D - return an error
    if len(data.shape) != 3:
        raise ValueError('Input data should be 3D.')

    data, epochs, channels, _ = check_dimensions_3D(data, n_epoch, n_ch)

    data_spat = np.array([spat_filter.T @ data[i, :channels] for i in range(epochs)])
    data_spat = data_spat * np.std(spat_pattern)

    return data_spat


def check_dimensions_3D(data, n_epoch, n_ch):
    """
    Check the dimensions of 3D array
    Should be epochs x channels x time

    :param numpy.ndarray data: data to check (dimensions - 3D)
    :param int n_ch: number of channels in the data
    :param int n_epoch: number of epochs
    :returns:
        data (numpy.ndarray, 2D) - data with dimensions epochs x channels x time;
        epochs (int) - number of epochs;
        channels (int) - number of channels;
        times (int) - number of time samples
    """
    (epochs, channels, times) = data.shape

    # check if first dimension is epochs
    if epochs != n_epoch:
        epoch_dim = np.where(np.array(data.shape) == n_epoch)[0][0]
        data = np.swapaxes(data, epoch_dim, 0)
        epochs = n_epoch
    # check if second dimension is channels
    if channels != n_ch:
        sen_dim = np.where(np.array(data.shape) == n_ch)[0][0]
        data = np.swapaxes(data, sen_dim, 1)
        channels = n_ch

    times = data.shape[2]

    return data, epochs, channels, times
