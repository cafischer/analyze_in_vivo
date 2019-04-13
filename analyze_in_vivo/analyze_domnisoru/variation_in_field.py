from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_track_len, get_last_bin_edge, \
    get_celltype_dict
from grid_cell_stimuli import get_AP_max_idxs
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import threshold_by_velocity, get_per_run, \
    get_bins_field_domnisoru
import matplotlib.gridspec as gridspec
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/variation_in_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes', 'fY_cm']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}

    # parameters
    use_AP_max_idxs_domnisoru = True
    bin_size = 5  # cm
    velocity_threshold = 1  # cm/sec

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    n_APs_cells = np.zeros(len(cell_ids), dtype=object)
    n_fields_cells = np.zeros(len(cell_ids), dtype=int)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        track_len = get_track_len(cell_id)

        # compute spike train
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        spike_train = get_spike_train(AP_max_idxs, len(v))

        # velocity threshold the data
        [v, t, position, spike_train], velocity = threshold_by_velocity([v, t, position, spike_train], velocity,
                                                                        velocity_threshold)

        # bin according to position and compute firing rate
        bins = np.arange(0, get_last_bin_edge(cell_id), bin_size)  # use same as matlab's edges

        # compute firing rate of original spike train
        # firing_rate = get_spatial_firing_rate(spike_train, position, bins, dt)
        # firing_rate_per_run = get_spatial_firing_rate_per_run(spike_train, position, bins, dt, track_len)

        bins_in_field = get_bins_field_domnisoru(cell_id, save_dir, bins)
        new_field = np.concatenate((np.array([0]), np.where(np.diff(bins_in_field) > 1)[0] + 1, np.array([len(bins_in_field)])))
        bins_in_field = [bins_in_field[i1:i2] for i1, i2 in zip(new_field[:-1], new_field[1:])]

        # TODO
        # pl.figure()
        # for run in range(np.size(firing_rate_per_run, 0)):
        #     pl.plot(bins[:-1], firing_rate_per_run[run, :])
        # pl.plot(bins[:-1], np.mean(firing_rate_per_run, 0), 'r')
        # pl.fill_between(bins[:-1],
        #                 np.mean(firing_rate_per_run, 0) - np.std(firing_rate_per_run, 0),
        #                 np.mean(firing_rate_per_run, 0) + np.std(firing_rate_per_run, 0), color='r', alpha=0.5)
        # pl.plot(bins[:-1], firing_rate, 'k', linewidth=1.5)
        # in_field_domnisoru[in_field_domnisoru == 0] = np.nan
        # pl.plot(bins[:-1], in_field_domnisoru * -1, 'g', linewidth=3.0)
        # pl.show()

        n_fields_cells[cell_idx] = len(bins_in_field)

        spike_train_runs = get_per_run(spike_train, position, get_track_len(cell_id))
        position_runs = get_per_run(position, position, get_track_len(cell_id))
        #v_runs = get_per_run(v, position, get_track_len(cell_id))
        n_runs = np.size(spike_train_runs, 0)
        n_APs = np.zeros((n_fields_cells[cell_idx], n_runs))

        for field in range(n_fields_cells[cell_idx]):
            for run in range(n_runs):
                position_binned_run = np.digitize(position_runs[run], bins) - 1
                in_field_run = np.array([True if p_b in bins_in_field[field] else False for p_b in position_binned_run])
                #v_run = v_runs[run][in_field_run]
                n_APs[field, run] = np.sum(spike_train_runs[run][in_field_run])

                # print n_APs[field, run]
                # pl.figure()
                # pl.plot(np.arange(len(v_run))*dt, v_run, 'k')
                # pl.show()

        n_APs_cells[cell_idx] = n_APs

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
        fig, axes = pl.subplots(1, n_fields_cells[cell_idx], figsize=(2.5*n_fields_cells[cell_idx], 5))
        colors = ['b', 'r', 'g', 'y', 'm']
        for field in range(n_fields_cells[cell_idx]):
            axes[field].set_title('Field %i' % field)
            axes[field].hist(n_APs[field, :], bins=np.arange(0, np.max(n_APs), 1), color='0.5', align='left')
            axes[field].set_xlabel('# APs')
            axes[field].set_ylabel('Frequency')
        pl.savefig(os.path.join(save_dir_cell, 'hist_n_APs_per_field.png'))
        #pl.show()
        pl.close('all')

    # def plot_hist(ax, cell_idx, n_APs_cells, n_fields_cells):
    #     colors = ['b', 'r', 'g', 'y', 'm', 'c', 'orange', 'k']
    #     for field in range(n_fields_cells[cell_idx]):
    #         ax.hist(n_APs_cells[cell_idx][field, :], bins=np.arange(0, np.max(n_APs_cells[cell_idx][field, :]), 1),
    #                 color=colors[field], align='left', alpha=0.5)
    #
    #
    # plot_kwargs = dict(n_APs_cells=n_APs_cells, n_fields_cells=n_fields_cells)
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_hist, plot_kwargs,
    #                         xlabel='# APs in field', ylabel='Frequency', sharey='none', sharex='none',
    #                         save_dir_img=os.path.join(save_dir_img, 'hist_n_APs_per_field.png'))
    #
    # def plot_hist(ax, cell_idx, n_APs_cells, n_fields_cells):
    #     colors = ['b', 'r', 'g', 'y', 'm', 'c', 'orange', 'k']
    #     for field in range(n_fields_cells[cell_idx]):
    #         ax.hist(n_APs_cells[cell_idx][field, :], bins=np.arange(0, np.max(n_APs_cells[cell_idx][field, :]), 1),
    #                 color=colors[field], align='left', alpha=0.5)
    #     ax.set_xlim(-0.5, 10.5)
    #
    #
    # plot_kwargs = dict(n_APs_cells=n_APs_cells, n_fields_cells=n_fields_cells)
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_hist, plot_kwargs,
    #                         xlabel='# APs in field', ylabel='Frequency', sharey='none', sharex='none',
    #                         save_dir_img=os.path.join(save_dir_img, 'hist_n_APs_per_field_max_10.png'))


    # n_rows = 3
    # n_columns = 9
    # fig = pl.figure(figsize=(14, 8.5))
    # outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.3, hspace=0.43)
    # cell_idx = 0
    # for i in range(n_rows * n_columns):
    #     inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i], hspace=0.2)
    #     if cell_idx < len(cell_ids):
    #         for field in range(4):
    #             if field < n_fields_cells[cell_idx] :
    #                 ax = pl.Subplot(fig, inner[field])
    #                 if field == 0:
    #                     if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
    #                         ax.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
    #                     elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
    #                         ax.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
    #                     else:
    #                         ax.set_title(cell_ids[cell_idx], fontsize=12)
    #
    #                 ax.hist(n_APs_cells[cell_idx][field, :], bins=np.arange(0, np.max(n_APs_cells[cell_idx]), 1),
    #                                color='0.5', align='left')
    #                 ax.xaxis.set_tick_params(labelsize=8)
    #                 ax.yaxis.set_tick_params(labelsize=8)
    #
    #                 if (i >= (n_rows - 1) * n_columns) and (field == 2 or field == 3):
    #                     ax.set_xlabel('# APs', fontsize=12)
    #                 if (i % n_columns == 0) and (field == 0 or field == 2):
    #                     ax.set_ylabel('Frequency', fontsize=12)
    #                 fig.add_subplot(ax)
    #         cell_idx += 1
    # pl.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.05)
    # pl.savefig(os.path.join(save_dir_img, 'hist_n_APs_per_field_subplots.png'))
    #
    # n_rows = 3
    # n_columns = 9
    # fig = pl.figure(figsize=(14, 8.5))
    # outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.3, hspace=0.43)
    # cell_idx = 0
    # for i in range(n_rows * n_columns):
    #     inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i], hspace=0.2)
    #     if cell_idx < len(cell_ids):
    #         for field in range(4):
    #             if field < n_fields_cells[cell_idx]:
    #                 ax = pl.Subplot(fig, inner[field])
    #                 if field == 0:
    #                     if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
    #                         ax.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
    #                     elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
    #                         ax.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
    #                     else:
    #                         ax.set_title(cell_ids[cell_idx], fontsize=12)
    #
    #                 ax.hist(n_APs_cells[cell_idx][field, :], bins=np.arange(0, np.max(n_APs_cells[cell_idx]), 1),
    #                         color='0.5', align='left')
    #                 ax.xaxis.set_tick_params(labelsize=8)
    #                 ax.yaxis.set_tick_params(labelsize=8)
    #                 ax.set_xlim(-0.5, 10.5)
    #
    #                 if (i >= (n_rows - 1) * n_columns) and (field == 2 or field == 3):
    #                     ax.set_xlabel('# APs', fontsize=12)
    #                 if (i % n_columns == 0) and (field == 0 or field == 2):
    #                     ax.set_ylabel('Frequency', fontsize=12)
    #                 fig.add_subplot(ax)
    #         cell_idx += 1
    # pl.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.05)
    # pl.savefig(os.path.join(save_dir_img, 'hist_n_APs_per_field_subplots_max_10.png'))
    # pl.show()