from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli import get_AP_max_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train, smooth_firing_rate
from grid_cell_stimuli.ISI_hist import get_ISIs
import scipy.stats as st
import warnings
import time
pl.style.use('paper_subplots')


def plot_autocorrelation(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    ax.set_xlim(-max_lag, max_lag)


def plot_autocorrelation_with_kde(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag, kernel_cells, dt_kernel=0.01):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    t_kernel = np.arange(-max_lag, max_lag+dt_kernel, dt_kernel)
    ax.plot(t_kernel, kernel_cells[cell_idx].pdf(t_kernel), color='k')
    ax.set_xlim(-max_lag, max_lag)


def get_crosscorrelation(x, y, max_lag=0):
    assert len(x) == len(y)
    cross_corr = np.zeros(2 * max_lag + 1)
    for lag in range(max_lag, 0, -1):
        cross_corr[max_lag - lag] = np.correlate(x[:-lag], y[lag:], mode='valid')[0]
    for lag in range(1, max_lag + 1, 1):
        cross_corr[max_lag + lag] = np.correlate(x[lag:], y[:-lag], mode='valid')[0]
        cross_corr[max_lag] = np.correlate(x, y, mode='valid')[0]

    assert np.all(cross_corr[:max_lag] == cross_corr[max_lag + 1:][::-1])
    return cross_corr


def get_autocorrelation(x, max_lag=50):
    auto_corr_lag = np.zeros(max_lag)
    for lag in range(1, max_lag+1, 1):
        auto_corr_lag[lag-1] = np.correlate(x[:-lag], x[lag:], mode='valid')[0]
    auto_corr_no_lag = np.array([np.correlate(x, x, mode='valid')[0]])
    auto_corr = np.concatenate((np.flipud(auto_corr_lag), auto_corr_no_lag, auto_corr_lag))
    return auto_corr


def get_autocorrelation_by_ISIs(ISIs, max_lag=50, bin_width=1, remove_zero=True, normalize=True):
    """
    Computes the autocorrelation of some spike train by means of the ISIs.
    :param ISIs: All ISIs obtained from the spike train. They need to be kept in the same order!
    :type ISIs: array
    :param max_lag: Up to which time the auto-correlation should be computed.
    :type max_lag: float
    :param bin_width: Width of the bins for the auto-correlation.
    :type bin_width: float
    :return: Spike-time autocorrelation, center of the bins, bins.
    """
    ISIs_cum = np.cumsum(ISIs)
    SIs = get_all_SIs(ISIs_cum)
    SIs = SIs[np.abs(SIs) <= max_lag]

    bins_half = np.arange(bin_width / 2., max_lag + bin_width / 2. + bin_width, bin_width)  # bins are centered
    bins_half += np.spacing(bins_half)  # without spacing the histogram would not be symmetric as comparison operation is different for both sides of the bin edges
    bins = np.concatenate((-bins_half[::-1], bins_half))
    autocorr = np.histogram(SIs, bins=bins)[0]
    t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

    # half_len = int((len(autocorr) - 1) / 2)
    # assert np.all(autocorr[half_len+1:] == autocorr[:half_len][::-1])  # control: autocorr must be symmetric

    if remove_zero:
        autocorr[max_lag_idx] = 0
    if normalize:
        autocorr = autocorr / (np.sum(autocorr) * bin_width)
    return autocorr, t_autocorr, bins


def get_all_SIs(ISIs_cum):
    ISIs_cum = np.insert(ISIs_cum, 0, 0)  # add distance to 1st spike
    SI_mat = np.tile(ISIs_cum, (len(ISIs_cum), 1)) - np.array([ISIs_cum]).T
    return SI_mat.flatten()  #np.array(all_SIs)bins[:int(len(bins/2.))]


# def change_bin_size_of_spike_train(spike_train, bin_size, dt):
#     bin_change = bin_size / dt
#     spike_train_new = np.zeros(int(round(len(spike_train) / bin_change)))
#     for i in range(len(spike_train_new)):
#         if sum(spike_train[i * int(bin_change):(i + 1) * int(bin_change)] == 1) == 1:
#             spike_train_new[i] = 1
#         elif sum(spike_train[i * int(bin_change):(i + 1) * int(bin_change)] == 1) == 0:
#             spike_train_new[i] = 0
#         else:
#             warnings.warn('More than one spike in bin!')
#     return spike_train_new


if __name__ == '__main__':
    save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)

    cell_type_dict = get_celltype_dict(save_dir)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    bin_width = 1.0  # ms
    max_lag = 50.0
    sigma_smooth = 1.0  # ms  None for no smoothing
    dt_kernel = 0.05
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)
    max_lag_idx = to_idx(max_lag, bin_width)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    autocorr_cells = np.zeros((len(cell_ids), int(2 * max_lag / bin_width + 1)))
    autocorr_kernel_cells = np.zeros((len(cell_ids), int(2 * max_lag / dt_kernel + 1)))
    kernel_cells = np.zeros(len(cell_ids), dtype=object)
    peak_autocorr = np.zeros(len(cell_ids))
    theta_power = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        # get ISIs
        ISIs = get_ISIs(AP_max_idxs, t)

        # get autocorrelation
        auto_corr, t_auto_corr, bins = get_autocorrelation_by_ISIs(ISIs, max_lag=max_lag, bin_width=bin_width)
        autocorr_cells[cell_idx, :] = auto_corr
        peak_autocorr[cell_idx] = t_auto_corr[max_lag_idx:][np.argmax(auto_corr[max_lag_idx:])]  # start at pos. half
                                                                                                  # as 1st max is taken

        # compute KDE
        ISIs_cum = np.cumsum(ISIs)
        SIs = get_all_SIs(ISIs_cum)
        SIs = SIs[np.abs(SIs) <= max_lag]
        SIs = SIs[SIs != 0]
        kernel = st.gaussian_kde(SIs, bw_method=np.sqrt(sigma_smooth/np.cov(SIs)))
        kernel_cells[cell_idx] = kernel
        t_kernel = np.arange(-max_lag, max_lag + dt_kernel, dt_kernel)
        autocorr_kernel_cells[cell_idx, :] = kernel_cells[cell_idx].pdf(t_kernel)

        # pl.figure()
        # pl.bar(t_auto_corr, auto_corr, bin_size, color='r', align='center', alpha=0.5)
        # t_kernel = np.arange(-50, 50+01, 0.1)
        # pl.plot(t_kernel, kernel.pdf(t_kernel), color='k')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # pl.show()

        # compute power in the theta range of FFT(auto_corr)
        smooth_auto_corr = smooth_firing_rate(auto_corr, std=1.0, window_size=3)
        fft_auto_corr = np.fft.fft(smooth_auto_corr)
        power = np.abs(fft_auto_corr)**2
        freqs = np.fft.fftfreq(smooth_auto_corr.size, d=(t_auto_corr[1] - t_auto_corr[0]) / 1000.0)
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        power = power[sort_idx]
        theta_power[cell_idx] = np.mean(power[np.logical_and(5 <= freqs, freqs <= 11)])

        # # plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        # print peak_auto_corr[cell_idx]
        # pl.close('all')
        # pl.figure()
        # pl.bar(t_auto_corr, auto_corr, bin_size, color='0.5', align='center')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # #pl.savefig(os.path.join(save_dir_cell, 'auto_corr_'+str(max_lag)+'.png'))
        # #pl.show()

        # pl.figure()
        # pl.plot(t_auto_corr, auto_corr, 'k')
        # pl.plot(t_auto_corr, smooth_auto_corr, 'r')
        # #pl.show()
        #
        # pl.figure()
        # pl.title('FFT auto-corr.')
        # pl.plot(freqs[sort_idx], power[sort_idx])
        # pl.xlim(0, 100)
        # pl.xlabel('Frequency (Hz)')
        # pl.ylabel('Power')
        # pl.show()

    # save autocorrelation
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'auto_corr_' + str(max_lag) + '_' + str(bin_width) + '_'+str(
            sigma_smooth) + '.npy'), autocorr_kernel_cells)
    else:
        np.save(os.path.join(save_dir_img, 'peak_auto_corr_' + str(max_lag) + '_' + str(bin_width) + '.npy'),
                peak_autocorr)
        np.save(os.path.join(save_dir_img, 'auto_corr_' + str(max_lag) + '_' + str(bin_width) + '.npy'), autocorr_cells)

    # plots
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    if sigma_smooth is not None:
        plot_kwargs = dict(t_auto_corr=t_auto_corr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
                           max_lag=max_lag, kernel_cells=kernel_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation_with_kde, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation',
                                save_dir_img=os.path.join(save_dir_img, 'auto_corr_' + str(max_lag) +'_' + str(
                                    bin_width) + '_' + str(sigma_smooth) + '.png'))
    else:
        plot_kwargs = dict(t_auto_corr=t_auto_corr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
                           max_lag=max_lag)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation',
                                save_dir_img=os.path.join(save_dir_img, 'auto_corr_' + str(max_lag) + '_' + str(
                                    bin_width) + '.png'))

    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
    #                         xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
    #                         wspace=0.18,
    #                         save_dir_img=os.path.join(save_dir_img2, 'auto_corr_' + str(max_lag) + '.png'))

    # # plot theta power
    # def plot_theta_power(ax, cell_idx, theta_power):
    #     ax.bar(0.5, theta_power[cell_idx], width=0.4, color='0.5')
    #     ax.set_xlim(0, 1)
    #     ax.set_xticks([])
    #
    # plot_kwargs = dict(theta_power=theta_power)
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_theta_power, plot_kwargs,
    #                         xlabel='', ylabel='Theta power',
    #                         save_dir_img=os.path.join(save_dir_img, 'theta_power.png'))
    pl.show()