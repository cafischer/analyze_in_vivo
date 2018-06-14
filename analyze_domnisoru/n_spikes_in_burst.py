from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones


def plot_n_spikes_in_burst_all_cells(count_spikes):
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                    axes[i1, i2].bar(bins[:-1],
                                     count_spikes[cell_idx, :] / (np.sum(count_spikes[cell_idx, :])*(bins[1]-bins[0])),
                                     color='0.5')
                    axes[i1, i2].set_xlim(bins[0]-0.5, bins[-1])
                    axes[i1, i2].set_xticks(bins)
                    labels = [''] * len(bins)
                    labels[::4] = bins[::4]
                    axes[i1, i2].set_xticklabels(labels)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('# Spikes \nin event')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Rel. Frequency')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'count_spikes.png'))

        if cell_type == 'grid_cells':
            n_rows = 3
            n_columns = 9
            fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
            cell_idx = 0
            for i1 in range(n_rows):
                for i2 in range(n_columns):
                    if cell_idx < len(cell_ids):
                        if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                            axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                        elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                            axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                        else:
                            axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                        axes[i1, i2].plot(bins[:-1],
                                         count_spikes[cell_idx, :] / (
                                                     np.sum(count_spikes[cell_idx, :]) * (bins[1] - bins[0])),
                                          marker='o', linestyle='-', color='0.5', markersize=5)

                        axes[i1, i2].set_yscale('log')
                        axes[i1, i2].set_xlim(bins[0] - 0.5, bins[-1])
                        axes[i1, i2].set_xticks(bins)
                        labels = [''] * len(bins)
                        labels[::4] = bins[::4]
                        axes[i1, i2].set_xticklabels(labels)
                        if i1 == (n_rows - 1):
                            axes[i1, i2].set_xlabel('# Spikes \nin event')
                        if i2 == (0):
                            axes[i1, i2].set_ylabel('Rel. Frequency')
                    else:
                        axes[i1, i2].spines['left'].set_visible(False)
                        axes[i1, i2].spines['bottom'].set_visible(False)
                        axes[i1, i2].set_xticks([])
                        axes[i1, i2].set_yticks([])
                    cell_idx += 1
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'count_spikes_log_scale.png'))


def get_n_spikes_in_burst(burst_ISI_bool):
    groups = np.split(burst_ISI_bool, np.where(np.abs(np.diff(burst_ISI_bool)) == 1)[0] + 1)
    burst_groups = []
    for g in groups:
        if True in g:
            burst_groups.append(g)
    return np.array([len(g) + 1 for g in burst_groups])


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/n_spikes_in_burst'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}

    bins = np.arange(1, 15 + 1, 1)
    ISI_burst = 10
    use_AP_max_idxs_domnisoru = True

    count_spikes = np.zeros((len(cell_ids), len(bins)-1))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

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

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_spikes_in_bursts = get_n_spikes_in_burst(short_ISI_indicator.astype(int))
        count_spikes[cell_idx, :] = np.histogram(n_spikes_in_bursts, bins)[0]

        starts_burst, ends_burst = get_start_end_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))
        count_spikes[cell_idx, 0] = len(AP_max_idxs_single)
        assert bins[0] == 1

        pl.close('all')
        pl.figure()
        pl.bar(bins[:-1], count_spikes[cell_idx, :], color='0.5')
        pl.xlabel('# Spikes')
        pl.ylabel('Frequency')
        pl.xticks(bins)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'n_spikes_in_burst.png'))
        #pl.show()

    # plot all cells
    pl.close('all')
    plot_n_spikes_in_burst_all_cells(count_spikes)
    pl.show()