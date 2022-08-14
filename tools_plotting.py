import numpy as np
import matplotlib.pyplot as plt


def topoplot_with_colorbar(data, raw_info, vmin=None, vmax=None, cmap='bone', mask=None):
    """
    Plot topography with a colorbar on the right side

    :param numpy.ndarray data: data to plot, should be size (channels,)
    :param mne.Info raw_info: instance of mne.Infor corresponding to the data
    :param float vmin: lower limit
    :param float vmax: upper limit
    :param cmap: either str or instance of LinearSegmentedColormap, colormap to use, default = 'bone'
    :param numpy.ndarray mask: mask corresponding the data, will be plotted in black 'x', default = None
    """
    import mne

    mask_params = dict(marker='x', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=4)

    # create a figure and plot topography
    fig, ax1 = plt.subplots(ncols=1)
    im, cm = mne.viz.plot_topomap(data, raw_info, vmin=vmin, vmax=vmax, ch_type='eeg', cmap=cmap, axes=ax1,
                                  show=False, show_names=False, mask=mask, mask_params=mask_params)
    # insert colorbar and adjust the position
    ax_x_start = 0.85
    ax_x_width = 0.04
    ax_y_start = 0.05
    ax_y_height = 0.85
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)


def parula_map():
    """
    Creates parula map with colors as in Matlab
    :returns:
    parula_map : instance of matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
               [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
               [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
               [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
               [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
               [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
               [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
               [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
               [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
               [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
               [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
               [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
               [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
               [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
               [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
               [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
               [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
               [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
               [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
               [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
               [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
               [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
               [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
               [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
               [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
               [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
               [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
               [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
               [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
               [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
               [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
               [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map


def plot_with_sem_one_line(x, y, ax, ax_num, xlim, ylim, color_y, color_y_sem, label_y=None, marker_y=None,
                           alpha_level=1, xlabel='Time, s', ylabel='Amplitude, V'):
    """
    Plots a line with standard error of the mean

    :param numpy.ndarray x: data on the x-axis
    :param numpy.ndarray y: corresponding data on the y-axis
    :param numpy.ndarray ax: axis handle
    :param int ax_num: number of the axis into which to plot
    :param list xlim: lower and upper limit for the x-axis
    :param list ylim: lower and upper limit for the y-axis
    :param str color_y: color of the line
    :param str color_y_sem: color of the sem-shadow
    :param str label_y: label to the legend, default = None
    :param str marker_y: marker to draw on the line, default = None
    :param float alpha_level: the alpha level for the sem transparency, default = 1
    :param str xlabel: label for x-axis, default = 'Time, s'
    :param str ylabel: label for y-axis, default = 'Amplitude, V'
    """
    from scipy.stats.stats import sem

    # plot the filling
    ax[ax_num].fill_between(x,
                            np.mean(y, axis=0) - sem(y, axis=0),
                            np.mean(y, axis=0) + sem(y, axis=0),
                            color=color_y_sem, alpha=alpha_level)
    # plot the line
    ax[ax_num].plot(x, np.mean(y, axis=0), marker=marker_y, color=color_y, label=label_y)

    ax[ax_num].legend()
    ax[ax_num].set_xlabel(xlabel)
    ax[ax_num].set_ylabel(ylabel)
    ax[ax_num].set_xlim(xlim)
    if ylim:
        ax[ax_num].set_ylim(ylim)


def plot_with_sem_color(x, y, ax, ax_num, xlim, color_y, color_y_sem, alpha_level, label_y, xlabel='Time, s',
                        ylabel='Amplitude, V'):
    """
    Plots several lines with standard error of the mean

    :param numpy.ndarray x: data on the x-axis
    :param list y: corresponding data on the y-axis, list os list or list of numpy.ndarray
    :param numpy.ndarray ax: axis handle
    :param int ax_num: number of the axis into which to plot
    :param list xlim: lower and upper limit for the x-axis
    :param list color_y: colors for each line
    :param list color_y_sem: colors of the sem-shadow for each line
    :param list alpha_level: the alpha level for each sem-shadow
    :param list label_y: labels of each line to the legend
    :param str xlabel: label for x-axis, default = 'Time, s'
    :param str ylabel: label for y-axis, default = 'Amplitude, V'

    """
    from scipy.stats.stats import sem

    # plot each sem and line in a loop
    if ax_num:
        for yi in range(len(y)):
            ax[ax_num].fill_between(x,
                                    np.mean(y[yi], axis=0) - sem(y[yi], axis=0),
                                    np.mean(y[yi], axis=0) + sem(y[yi], axis=0),
                                    color=color_y_sem[yi], alpha=alpha_level[yi])
            ax[ax_num].plot(x, np.mean(y[yi], axis=0), color=color_y[yi], label=label_y[yi])

        ax[ax_num].legend()
        ax[ax_num].set_xlabel(xlabel)
        ax[ax_num].set_ylabel(ylabel)
        ax[ax_num].set_xlim(xlim)
    else:
        for yi in range(len(y)):
            ax.fill_between(x, np.mean(y[yi], axis=0) - sem(y[yi], axis=0),
                               np.mean(y[yi], axis=0) + sem(y[yi], axis=0),
                               color=color_y_sem[yi], alpha=alpha_level[yi])
            ax.plot(x, np.mean(y[yi], axis=0), color=color_y[yi], label=label_y[yi])

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)


def annotated_heatmap(x_labels, y_labels, data, cbarlabel, cmap='winter'):
    """
    Plots heatmap with numbers

    The function is adapted from matplotlib documentation
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    :param list x_labels: list of labels for each cell on the x-axis
    :param list y_labels: list of labels for each cell on the x-axis
    :param numpy.ndarray data: data to plot (dimensions - 2D)
    :param str cbarlabel: label for a colorbar
    :param cmap: either str or instance of LinearSegmentedColormap, colormap to use, default = 'winter'

    """
    # create a figure and plot the data
    fig, ax = plt.subplots()
    vmin = -np.max(np.abs(data))
    vmax = np.max(np.abs(data))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # add text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="k", fontsize=12)
    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=90, va="bottom")
    fig.tight_layout()
    plt.show()


def parula_map_backward():
    """Creates parula map with colors as in Matlab but inverted (the blue is the highest values)
    :returns:
    parula_map : instance of matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
               [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
               [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
               [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
               [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
               [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
               [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
               [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
               [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
               [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
               [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
               [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
               [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
               [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
               [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
               [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
               [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
               [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
               [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
               [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
               [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
               [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
               [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
               [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
               [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
               [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
               [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
               [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
               [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
               [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
               [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
               [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]][::-1]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map
