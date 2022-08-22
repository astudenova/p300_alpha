"""
This code creates simulated data for fig in Methods.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

def create_signal_nzm(n_samples, fs, nzm_coeff):
    """
    Generates alpha signals with non-zero mean profile
    
    :param int n_samples: the length of generated signal
    :param int fs: sampling rate
    :param float nzm_coeff: non-zero mean value, if "-" then neg mean, if "+" then pos mean, 
        if "0" = zero mean oscillations
    :return: signal (numpy.ndarray, 1D) - simulated signal
    """

    x1 = 2.5 * np.random.randn(n_samples)
    b10, a10 = butter(N=2, Wn=np.array([8, 12]) / fs * 2, btype='bandpass')
    alpha = filtfilt(b10, a10, x1)

    # if - then neg mean, if + then pos mean
    signal = alpha + nzm_coeff * np.abs(alpha) + 0.01 * np.random.randn(n_samples)
    return signal


def plot_curves(x_data, y_data, colors, xlim, ylim, linewidth=1):
    """
    Plot identical plots for the figure

    :param numpy.ndarray x_data: data on the x-axis (dimensions - 1D)
    :param numpy.ndarray y_data: data on the y-axis (dimensions - 1D or 2D)
    :param colors: colors to plot the curves with
    :param list xlim: limits for the x-axis
    :param list ylim: limits for the y-axis
    :param int linewidth: the thickness of the lines, default = 1
    """
    # check data dimensions
    n_times = len(x_data)
    if len(y_data.shape) > 2:
        raise ValueError('Input data should be 1D or 2D.')

    if len(y_data.shape) == 1:
        num_lines = 1
        if y_data.shape[0] != n_times:
            raise ValueError('The length of y_data should be the same as length of x_data.')

    if len(y_data.shape) == 2:
        if n_times != y_data.shape[1]:
            y_data = y_data.T
            num_lines = y_data.shape[0]
        else:
            num_lines = y_data.shape[0]

    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Time, s')
    plt.ylabel('Amplitude, a.u.')
    if num_lines == 1:
        plt.plot(x_data, y_data, c=colors, linewidth=linewidth)
    else:
        for i, color in enumerate(colors):
            plt.plot(x_data, y_data[i], c=color, linewidth=linewidth)
    # zero line y = 0
    plt.plot(x_data, np.zeros(x_data.shape), c='grey', alpha=.5)
    # zero line x = 0
    plt.vlines(0, ymin=-ylim[0], ymax=ylim[1], colors='grey', alpha=.5, linestyles='--')



n = 2500
fs = 1000
x = np.linspace(-.8, 1.7, int(2.5 * fs))
n_epoch = 100

# create amplitude modulation
peaktime = .35
peak_time_smp = np.argmin(np.abs(x - peaktime))
width1 = .1
width2 = .35
gauss1 = 0.2 + np.ones((n,)) - np.exp(-(x - peaktime) ** 2 / (2 * width1 ** 2))
gauss2 = 0.2 + np.ones((n,)) - np.exp(-(x - peaktime) ** 2 / (2 * width2 ** 2))
amplMod = np.hstack((gauss1[:peak_time_smp], gauss2[peak_time_smp:]))
amplMod = amplMod / np.max(amplMod)
# create epochs
y = np.zeros((n_epoch, n))
for ni in range(n_epoch):
    alpha = create_signal_nzm(n, fs, nzm_coeff=-0.4).reshape((-1))
    y[ni] = np.multiply(alpha, amplMod)

# pic 1a
plot_curves(x, y[0], colors='darkblue', xlim=[-.5, 1.2], ylim=[-.9, .9])

# pic 1c
colors = ['darkblue', 'royalblue', 'blue', 'steelblue', 'dodgerblue', 'cornflowerblue']
plot_curves(x, y[:len(colors)], colors=colors, xlim=[-.5, 1.2], ylim=[-.9, .9])

# pic 1b
b10, a10 = butter(N=2, Wn=np.array([8, 12]) / 1000 * 2, btype='bandpass')
y_filt = filtfilt(b10, a10, y, axis=1)
y_env = np.abs(hilbert(y_filt, axis=1))
plot_curves(x, np.vstack((y[0], y_filt[0], y_env[0])),
            colors=['darkblue', 'orange', 'goldenrod'], xlim=[-.5, 1.2], ylim=[-.9, .9])

# pic 1d
colors = ['goldenrod', 'orange', 'darkgoldenrod', 'darkorange', 'gold', 'tan']
plot_curves(x, y_env[:len(colors)], colors=colors, xlim=[-.5, 1.2], ylim=[-.9, .9])

# pic 1e
lf = 3
b_lf, a_lf = butter(N=4, Wn=lf / 1000 * 2, btype='lowpass')
erp = filtfilt(b_lf, a_lf, np.mean(y, axis=0)) - \
      np.mean(filtfilt(b_lf, a_lf, np.mean(y, axis=0)[500:700]))
plot_curves(x, erp, colors='darkblue', xlim=[-.5, 1.2], ylim=[-.09, .09], linewidth=5)

# pic 1f
plot_curves(x, np.mean(y_env, axis=0), colors='goldenrod',
            xlim=[-.5, 1.2], ylim=[-.3, .3], linewidth=5)
