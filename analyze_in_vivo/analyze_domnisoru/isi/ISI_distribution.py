from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from scipy.stats import ks_2samp
from itertools import combinations
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
pl.style.use('paper')

if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True

    # parameter
    max_ISIs = np.arange(5, 100, 5)  #[5, 10, 15, 20, 50, 100]
    bin_width = 2.0

    # over cells
    std_ISI_cells = np.zeros((len(cell_ids), len(max_ISIs)))
    mean_ISI_cells = np.zeros((len(cell_ids), len(max_ISIs)))

    for cell_idx, cell_id in enumerate(cell_ids):
        for max_ISI_idx, max_ISI in enumerate(max_ISIs):
            bins = np.arange(0, max_ISI + bin_width, bin_width)
            print cell_id
            # load
            data = load_data(cell_id, param_list, save_dir)
            v = data['Vm_ljpc']
            t = np.arange(0, len(v)) * data['dt']
            dt = t[1] - t[0]

            # ISIs
            if use_AP_max_idxs_domnisoru:
                AP_max_idxs = data['spiketimes']
            else:
                AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
            ISIs = get_ISIs(AP_max_idxs, t)
            ISIs = ISIs[ISIs <= max_ISI]

            # mean and std
            std_ISI_cells[cell_idx, max_ISI_idx] = np.std(ISIs)
            mean_ISI_cells[cell_idx, max_ISI_idx] = np.mean(ISIs)

    # plots
    # n_rows = 2
    # n_cols = 3
    # fig, axes = pl.subplots(n_rows, n_cols, figsize=(10, 6))
    # max_ISI_idx = 0
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         axes[i, j].set_title('cut at %i (ms)' % max_ISIs[max_ISI_idx])
    #         axes[i, j].plot(std_ISI_cells[:, max_ISI_idx], mean_ISI_cells[:, max_ISI_idx], 'ok', markersize=3)
    #         max_ISI_idx += 1
    #         if j == 0:
    #             axes[i, j].set_ylabel('Mean ISI (ms)')
    #         if i == (n_rows - 1):
    #             axes[i, j].set_xlabel('Std ISI (ms)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, cell_type, 'std_vs_mean_ISI.png'))
    # #pl.show()

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

                    axes[i1, i2].plot(max_ISIs, mean_ISI_cells[cell_idx, :], 'ok', markersize=3)

                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('Max. ISI dur. (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Mean ISI (ms)')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, cell_type, 'mean_ISI.png'))

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

                    axes[i1, i2].plot(max_ISIs, std_ISI_cells[cell_idx, :], 'ok', markersize=3)

                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('Max. ISI dur. (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Std ISI (ms)')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, cell_type, 'std_ISI.png'))
        pl.show()