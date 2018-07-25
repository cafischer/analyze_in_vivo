from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype, get_last_bin_edge
import scipy.signal
from cell_fitting.util import init_nan
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec


def threshold_by_velocity(arrays_to_shorten, velocity, threshold=1):
    """
    Remove regions where velocity < threshold from the data. Note: Data will contain discontinuities.
    :param arrays_to_shorten:
    :type list[array]
    :param velocity: (cm/sec).
    :param threshold: Threshold (cm/sec) below which to cut out the data.
    :return: arrays_to_shorten, velocity with regions removed where velocity < threshold.
    """
    to_low = velocity < threshold
    for i in range(len(arrays_to_shorten)):
        arrays_to_shorten[i] = arrays_to_shorten[i][~to_low]
    velocity = velocity[~to_low]
    return arrays_to_shorten, velocity


def get_spike_train(AP_max_idxs, len_v):
    spike_train = np.zeros(len_v)
    spike_train[AP_max_idxs] = 1
    return spike_train


def get_spatial_firing_rate_per_run(spike_train, position, bins, dt, track_len):
    """
    Computes the spatial firing rate per run for the given bins.
    :param spike_train: Spike train.
    :param t: Time.
    :param position: Position of the animal on the track.
    :param bins: Bins for the position of the animal.
    :return: firing_rate_per_run: Spatial firing rate for each run through the track.
    """
    run_split_idxs = np.where(np.diff(position) < -track_len / 2.)[0] + 1  # +1 because diff shifts one to front
    spike_train_runs = np.split(spike_train, run_split_idxs)
    position_runs = np.split(position, run_split_idxs)
    n_bins = len(bins) - 1  # -1 for last edge

    firing_rate_per_run = init_nan((len(run_split_idxs) + 1, n_bins))
    for i_run, (spike_train_run, position_run) in enumerate(zip(spike_train_runs, position_runs)):
        firing_rate_per_run[i_run, :] = get_spatial_firing_rate(spike_train_run, position_run, bins, dt)
    return firing_rate_per_run


def get_spatial_firing_rate(spike_train, position, bins, dt):
    """
    Computes the spatial firing rate for the given bins.
    :param spike_train: Spike train.
    :param position: Position of the animal on the track.
    :param bins: Bins for the position of the animal.
    :return: firing_rate: Spatial firing rate.
    """
    AP_count_per_bin = np.histogram(position[spike_train.astype(bool)], bins=bins)[0]
    seconds_per_bin = np.histogram(position, bins=bins)[0] * dt / 1000.0
    firing_rate = init_nan(len(bins)-1)
    firing_rate[seconds_per_bin == 0] = np.nan
    firing_rate[seconds_per_bin != 0] = AP_count_per_bin[seconds_per_bin != 0] / seconds_per_bin[seconds_per_bin != 0]
    firing_rate[np.isnan(firing_rate)] = 0  # TODO: this is what Domnisoru does
    return firing_rate


def get_spatial_occupancy(position, bins, bin_size):
    occupancy_per_bin = np.histogram(position, bins=bins)[0]
    occupancy_per_bin = occupancy_per_bin / float(np.sum(occupancy_per_bin) * bin_size)
    return occupancy_per_bin


def smooth_firing_rate(firing_rate, std=1, window_size=3):
    window = scipy.signal.gaussian(window_size, std)
    window /= np.sum(window)

    not_nan = ~np.isnan(firing_rate)
    assert np.all(np.arange(np.where(not_nan)[0][0], np.where(not_nan)[0][-1]+1, 1)
                  == np.where(not_nan)[0])  # firing rate should only be nan at the edges (= ascending without holes in between)

    firing_rate_smoothed = init_nan(len(firing_rate))
    firing_rate_smoothed[not_nan] = np.convolve(firing_rate[not_nan], window, mode='same')  # with 0-padding

    # # for test: smoothed firing rate
    # pl.figure()
    # pl.plot(firing_rate, label='normal')
    # pl.plot(firing_rate_smoothed, label='smoothed')
    # pl.legend()
    # pl.show()
    return firing_rate_smoothed


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/position_vs_firing_rate'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_ids = ['s109_0002']
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}

    # parameters
    use_AP_max_idxs_domnisoru = True
    use_velocity_threshold = True
    bin_size = 4.0  # cm
    velocity_threshold = 1  # cm/sec

    if use_velocity_threshold:
        save_dir_img = os.path.join(save_dir_img, 'vel_thresh_'+str(velocity_threshold))
    save_dir_img = os.path.join(save_dir_img, cell_type, 'bin_size_' + str(bin_size))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    firing_rate_cells = []
    position_cells = []
    spike_train_cells = []
    t_cells = []
    bins_cells = []

    for cell_id in cell_ids:
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        bins = np.arange(0, get_last_bin_edge(cell_id), bin_size)  # use same as matlab's edges
        n_bins = len(bins) - 1  # -1 for last edge
        bins_cells.append(bins)

        # compute spike train
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        spike_train = get_spike_train(AP_max_idxs, len(v))

        # velocity threshold the data
        if use_velocity_threshold:
            [v, t, position, spike_train], velocity = threshold_by_velocity([v, t, position, spike_train], velocity,
                                                                            velocity_threshold)

        # compute firing rate
        firing_rate = smooth_firing_rate(get_spatial_firing_rate(spike_train, position, bins, dt))
        occupancy_prob = get_spatial_occupancy(position, bins, bin_size)

        # save
        np.save(os.path.join(save_dir_cell, 'avg_firing_rate.npy'), np.nanmean(firing_rate))
        np.save(os.path.join(save_dir_cell, 'firing_rate.npy'), firing_rate)
        np.save(os.path.join(save_dir_cell, 'position.npy'), bins[:-1])
        np.save(os.path.join(save_dir_cell, 'occupancy_prob.npy'), occupancy_prob)

        # save for later use
        firing_rate_cells.append(firing_rate)
        position_cells.append(position)
        t_cells.append(t)
        spike_train_cells.append(spike_train)

        # plot
        pl.close('all')
        fig, axes = pl.subplots(2, 1, sharex='all', figsize=(11, 6))
        # axes[0].plot(position, t / 1000., '0.5', linewidth=0.8)
        # axes[0].plot(position[spike_train.astype(bool)], (t / 1000.)[spike_train.astype(bool)], 'or', markersize=0.5)
        t_new = np.arange(len(position)) * dt / 1000.
        axes[0].plot(position, t_new, '0.5', linewidth=0.8)
        axes[0].plot(position[spike_train.astype(bool)],
                 t_new[spike_train.astype(bool)], 'or',
                 markersize=4.0)
        axes[0].set_ylabel('Time (s)')
        axes[1].plot(bins[:-1], firing_rate, 'k')
        axes[1].set_ylabel('Firing rate (Hz)')
        axes[1].set_xlabel('Position (cm)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'position_vs_firing_rate.png'))


    # plot all firing rates
    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.3, hspace=0.43)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.2)
            ax1 = pl.Subplot(fig, inner[0])
            ax2 = pl.Subplot(fig, inner[1])
            if cell_idx < len(cell_ids):
                if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)
                t_new = np.arange(len(position_cells[cell_idx]))*dt/1000.
                ax1.plot(position_cells[cell_idx], t_new, '0.5', linewidth=0.8)
                ax1.plot(position_cells[cell_idx][spike_train_cells[cell_idx].astype(bool)],
                         t_new[spike_train_cells[cell_idx].astype(bool)], 'or', markersize=0.1)
                ax1.set_xticklabels([])
                ax1.xaxis.set_tick_params(labelsize=10)
                ax2.xaxis.set_tick_params(labelsize=10)
                ax1.yaxis.set_tick_params(labelsize=10)
                ax2.yaxis.set_tick_params(labelsize=10)
                ax2.plot(bins_cells[cell_idx][:-1], firing_rate_cells[cell_idx], 'k')

                if i >= (n_rows - 1) * n_columns:
                    ax2.set_xlabel('Position \n(cm)')
                if i % n_columns == 0:
                    ax1.set_ylabel('Time (s)')
                    ax2.set_ylabel('Firing \nrate (Hz)')
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            cell_idx += 1
        pl.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.95)
        pl.savefig(os.path.join(save_dir_img, 'position_vs_firing_rate.png'))
        pl.show()

    else:
        n_rows = 1 if len(cell_ids) <= 3 else 2
        n_columns = int(round(len(cell_ids) / n_rows))
        fig_height = 4.5 if len(cell_ids) <= 3 else 9

        fig = pl.figure(figsize=(14, fig_height))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.3, hspace=0.4)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.2)
            ax1 = pl.Subplot(fig, inner[0])
            ax2 = pl.Subplot(fig, inner[1])
            if cell_idx < len(cell_ids):
                if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.plot(position_cells[cell_idx], t_cells[cell_idx] / 1000., '0.5', linewidth=0.8)
                ax1.plot(position_cells[cell_idx][spike_train_cells[cell_idx].astype(bool)],
                         (t_cells[cell_idx] / 1000.)[spike_train_cells[cell_idx].astype(bool)], 'or',
                         markersize=0.1)
                ax1.set_xticklabels([])
                ax2.plot(bins_cells[cell_idx][:-1], firing_rate_cells[cell_idx], 'k')

                if i >= (n_rows - 1) * n_columns:
                    ax2.set_xlabel('Position \n(cm)')
                if i % n_columns == 0:
                    ax1.set_ylabel('Time (s)')
                    ax2.set_ylabel('Firing \nrate (Hz)')
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            cell_idx += 1
        pl.subplots_adjust(left=0.07, right=0.98, top=0.95)
        pl.savefig(os.path.join(save_dir_img, 'position_vs_firing_rate.png'))
        pl.show()