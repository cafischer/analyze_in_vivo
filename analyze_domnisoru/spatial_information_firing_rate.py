from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
pl.style.use('paper')


def get_MI(x_binned, y_binned, bins_x, bins_y, n_bins_x, n_bins_y):
    prob_x_and_y = np.zeros((n_bins_x, n_bins_y))
    prob_x_times_y = np.zeros((n_bins_x, n_bins_y))
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            prob_x_and_y[i, j] = np.sum(np.logical_and(x_binned == i, y_binned == j)) / float(len(x_binned))
            prob_x_times_y[i, j] = np.sum(x_binned == i) / float(len(x_binned)) \
                                   * np.sum(y_binned == j) / float(len(x_binned))
    prob_x_and_y_flat = prob_x_and_y.flatten()
    prob_x_times_y_flat = prob_x_times_y.flatten()
    summands = np.zeros((n_bins_x * n_bins_y))
    not_zero = np.logical_and(prob_x_and_y_flat != 0, prob_x_times_y_flat != 0)
    summands[not_zero] = prob_x_and_y_flat[not_zero] \
                         * (np.log(prob_x_and_y_flat[not_zero] / prob_x_times_y_flat[not_zero]) / np.log(2))

    # plots for testing
    # x, y = np.meshgrid(bins_y, bins_x)
    # pl.figure()
    # pl.pcolor(x, y, prob_x_and_y)
    # pl.colorbar()
    # pl.figure()
    # pl.pcolor(x, y, prob_x_times_y)
    # pl.colorbar()
    # pl.show()

    return np.sum(summands)


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_firing_rate = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir_rec_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/recording_info'
    save_dir_fields = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'Y_cm']

    # parameters
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    n_APs = np.load(os.path.join(save_dir_rec_info, cell_type, 'n_APs.npy'))
    n_runs = np.load(os.path.join(save_dir_rec_info, cell_type, 'n_runs.npy'))
    n_fields = np.load(os.path.join(save_dir_fields, cell_type, 'n_fields.npy'))

    inv_entropy = np.zeros(len(cell_ids))
    spatial_info_skaggs = np.zeros(len(cell_ids))
    # MI_v = np.zeros(len(cell_ids))
    # MI_spiketrain = np.zeros(len(cell_ids))
    position_cells = []
    firing_rate_cells = []
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        pos = data['Y_cm']
        dt = t[1] - t[0]
        firing_rate = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'firing_rate.npy'))
        avg_firing_rate = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'avg_firing_rate.npy'))
        position = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'position.npy'))
        occupancy_prob = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'occupancy_prob.npy'))
        firing_rate_cells.append(firing_rate)
        position_cells.append(position)
        firing_rate_not_nan = firing_rate[~np.isnan(firing_rate)]
        bin_size_position = position[1] - position[0]

        # entropy
        prob_position = firing_rate_not_nan / (np.sum(firing_rate_not_nan) * bin_size_position)
        summands = np.zeros(len(firing_rate_not_nan))
        summands[prob_position != 0] = prob_position[prob_position != 0] * (np.log(prob_position[prob_position != 0])
                                                                             / np.log(2))  # by definition: if prob=0,
                                                                                           # then summand = 0
        entropy_cell = -np.sum(summands)

        prob_position_uniform = np.ones(len(position)) / (len(position) * bin_size_position)
        summands = prob_position_uniform * (np.log(prob_position_uniform) / np.log(2))
        entropy_uniform = -np.sum(summands)
        inv_entropy[cell_idx] = 1 - entropy_cell / entropy_uniform

        # Skaggs, 1996 spatial information
        scaled_firing_rate = firing_rate_not_nan / avg_firing_rate
        occupancy_prob_not_nan = occupancy_prob[~np.isnan(firing_rate)]
        summands = np.zeros(len(firing_rate_not_nan))
        summands[scaled_firing_rate != 0] = occupancy_prob_not_nan[scaled_firing_rate != 0] \
                                           * scaled_firing_rate[scaled_firing_rate != 0] \
                                           * (np.log(scaled_firing_rate[scaled_firing_rate != 0]) / np.log(2))
        spatial_info_skaggs[cell_idx] = np.sum(summands)

        # # Mutual information: voltage and position
        # bins_v = np.arange(-90, 20, 10)  # bin v
        # n_bins_v = len(bins_v) - 1
        # v_binned = np.digitize(v, bins_v) - 1
        # bin_size = 5  # cm
        # track_len = 400  # cm
        # bins_pos = np.arange(0, track_len + bin_size, bin_size)  # bin pos
        # n_bins_pos = len(bins_pos) - 1
        # pos_binned = np.digitize(pos, bins_pos) - 1
        # MI_v[cell_idx] = get_MI(v_binned, pos_binned, bins_v, bins_pos, n_bins_v, n_bins_pos)

        # # Mutual information: spiketrain and position
        # if use_AP_max_idxs_domnisoru:
        #     AP_max_idxs = data['spiketimes']
        # else:
        #     AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        # spike_train = get_spike_train(AP_max_idxs, len(v))
        # MI_spiketrain[cell_idx] = get_MI(spike_train, pos_binned, np.array([0, 1, 2]), bins_pos,
        #                                  2, n_bins_pos)

    np.save(os.path.join(save_dir_img, 'spatial_info.npy'), spatial_info_skaggs)

    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=4.0)
            ax1 = pl.Subplot(fig, inner[0])
            if cell_idx < len(cell_ids):
                if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.plot(position_cells[cell_idx], firing_rate_cells[cell_idx], 'k')
                ax1.annotate('Skaggs, 1996: %.2f' % spatial_info_skaggs[cell_idx],
                             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                             bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))
                if i >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('Position (cm)', fontsize=10)
                if i % n_columns == 0:
                    ax1.set_ylabel('Firing rate (Hz)')
                fig.add_subplot(ax1)
                cell_idx += 1
        pl.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        pl.savefig(os.path.join(save_dir_img, 'firing_rate_spatial_info.png'))
        #pl.show()

    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=4.0)
            ax1 = pl.Subplot(fig, inner[0])
            if cell_idx < len(cell_ids):
                if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.bar(1, (spatial_info_skaggs[cell_idx]), width=0.8, color='0.5')

                ax1.set_xlim(0, 2)
                ax1.set_ylim(np.min(spatial_info_skaggs), np.max(spatial_info_skaggs))
                ax1.set_xticks([])

                # if i >= (n_rows - 1) * n_columns:
                #     ax1.set_xlabel('Skaggs, 1996', fontsize=12)
                if i % n_columns == 0:
                    ax1.set_ylabel('Spatial information')
                fig.add_subplot(ax1)
                cell_idx += 1
        pl.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        pl.savefig(os.path.join(save_dir_img, 'spatial_info.png'))
        #pl.show()

        # fig = pl.figure(figsize=(14, 8.5))
        # outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)
        # cell_idx = 0
        # for i in range(n_rows * n_columns):
        #     inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=4.0)
        #     ax1 = pl.Subplot(fig, inner[0])
        #     if cell_idx < len(cell_ids):
        #         if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
        #             ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
        #         elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
        #             ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
        #         else:
        #             ax1.set_title(cell_ids[cell_idx], fontsize=12)
        #
        #         ax1.bar(1, (MI_v[cell_idx]), width=0.8, color='0.5')
        #
        #         ax1.set_xlim(0, 2)
        #         ax1.set_ylim(0.0, 1.0)
        #         ax1.set_xticks([])
        #
        #         if i >= (n_rows - 1) * n_columns:
        #             ax1.set_xlabel('MI', fontsize=12)
        #         if i % n_columns == 0:
        #             ax1.set_ylabel('Spatial information')
        #         fig.add_subplot(ax1)
        #         cell_idx += 1
        # pl.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        # pl.savefig(os.path.join(save_dir_img, 'MI_v.png'))

        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)
        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=4.0)
            ax1 = pl.Subplot(fig, inner[0])
            if cell_idx < len(cell_ids):
                if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.bar(1, (inv_entropy[cell_idx]), width=0.8, color='0.5')

                ax1.set_xlim(0, 2)
                ax1.set_ylim(0.0, 1.0)
                ax1.set_xticks([])

                if i >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('Inv. entropy', fontsize=12)
                if i % n_columns == 0:
                    ax1.set_ylabel('Spatial information')
                fig.add_subplot(ax1)
                cell_idx += 1
        pl.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        pl.savefig(os.path.join(save_dir_img, 'inv_entropy.png'))
        # pl.show()

        pl.figure()
        pl.plot(inv_entropy, spatial_info_skaggs, 'ok')
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('Inv. entropy')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'entropy_vs_skaggs.png'))

        # pl.figure()
        # pl.plot(MI_v, spatial_info_skaggs, 'ok')
        # pl.ylabel('Skaggs, 1996')
        # pl.xlabel('MI (mem. pot.)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'mi_vs_skaggs.png'))

        # plot dependence of Skaggs, 1996 measure on firing characteristics
        fig, ax = pl.subplots()
        plot_with_markers(ax, n_APs, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('# APs')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_APs_vs_skaggs.png'))

        fig, ax = pl.subplots()
        plot_with_markers(ax, n_runs, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('# Runs')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_runs_vs_skaggs.png'))

        fig, ax = pl.subplots()
        pl.plot(n_fields, spatial_info_skaggs, 'ok')
        plot_with_markers(ax, n_fields, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('# Fields')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_fields_vs_skaggs.png'))

        pl.show()