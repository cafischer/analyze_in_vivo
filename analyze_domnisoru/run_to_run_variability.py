from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import threshold_by_velocity, get_spike_train


def get_n_APs_per_run(spike_train, position, track_len):
    run_start_idxs = np.where(np.diff(position) < -track_len/2.)[0] + 1  # +1 because diff shifts one to front
    spike_train_runs = np.split(spike_train, run_start_idxs)
    n_APs_per_run = np.zeros(len(spike_train_runs))
    for i, spike_train_run in enumerate(spike_train_runs):
        n_APs_per_run[i] = np.sum(spike_train_run)
    return n_APs_per_run


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50,
                     's76_0002': -50, 's101_0009': -45}

    # parameters
    use_AP_max_idxs_domnisoru = True
    use_velocity_threshold = True
    bin_size = 5  # cm
    track_len = 400  # cm
    velocity_threshold = 1  # cm/sec

    if use_velocity_threshold:
        save_dir_img = os.path.join(save_dir_img, 'vel_thresh_'+str(velocity_threshold))
    bins = np.arange(0, track_len + bin_size, bin_size)
    n_bins = len(bins) - 1  # -1 for last edge
    bins_n_APs = np.arange(0, 400, 2)

    n_APs_per_run_cells = []
    n_APs_per_run_mean = np.zeros(len(cell_ids))
    n_APs_per_run_std = np.zeros(len(cell_ids))
    n_APs_per_run_cv = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]

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
        n_APs_per_run = get_n_APs_per_run(spike_train, position, track_len)
        n_APs_per_run_mean[cell_idx] = np.mean(n_APs_per_run)
        n_APs_per_run_std[cell_idx] = np.std(n_APs_per_run)
        n_APs_per_run_cv[cell_idx] = n_APs_per_run_std[cell_idx] / n_APs_per_run_mean[cell_idx]

        # save for later use
        n_APs_per_run_cells.append(n_APs_per_run)

        # plot
        pl.close('all')
        fig, axes = pl.subplots(1, 1)
        axes.hist(n_APs_per_run, bins, color='0.5')
        axes.annotate('Mean: %.2f \nStd: %.2f \nCV: %.2f'
                      % (n_APs_per_run_mean[cell_idx], n_APs_per_run_std[cell_idx], n_APs_per_run_cv[cell_idx]),
                      xy=(0.75, 0.9), xycoords='figure fraction', fontsize=14, ha='left', va='top',
                      bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))
        axes.set_ylabel('# Runs')
        axes.set_xlabel('# APs')
        axes.set_xlim(0, bins_n_APs[-1])
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'n_APs_per_run.png'))
        #pl.show()

    # plot all
    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.3, hspace=0.2)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i])
            ax1 = pl.Subplot(fig, inner[0])
            if cell_idx < len(cell_ids):
                if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.hist(n_APs_per_run_cells[cell_idx], bins, color='0.5')
                ax1.annotate('Mean: %.2f \nStd: %.2f \nCV: %.2f'
                              % (n_APs_per_run_mean[cell_idx], n_APs_per_run_std[cell_idx], n_APs_per_run_cv[cell_idx]),
                              xy=(0.35, 0.9), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                              bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))
                ax1.xaxis.set_tick_params(labelsize=10)
                ax1.yaxis.set_tick_params(labelsize=10)

                if i >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('# APs')
                else:
                    ax1.set_xticklabels([])
                if i % n_columns == 0:
                    ax1.set_ylabel('# Runs')
            fig.add_subplot(ax1)
            cell_idx += 1
        pl.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.06)
        pl.savefig(os.path.join(save_dir_img, cell_type, 'n_APs_per_run.png'))
        pl.show()