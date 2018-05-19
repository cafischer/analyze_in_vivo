from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs


def plot_n_spikes_in_burst_all_cells(n_spikes_in_burst_groups_per_cell):
    n_rows = 1 if len(cell_ids) <= 3 else 2
    n_columns = int(round(len(cell_ids) / n_rows))
    fig_height = 4.5 if len(cell_ids) <= 3 else 9
    fig, axes = pl.subplots(n_rows, n_columns, sharex='all', figsize=(14, fig_height))
    if n_rows == 1:
        axes = np.array([axes])
    if len(cell_ids) == 1:
        axes = np.array([axes])
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(int(round(len(cell_ids) / n_rows))):
            if cell_idx < len(cell_ids):
                axes[i1, i2].hist(n_spikes_in_burst_groups_per_cell[cell_idx], bins=bins, color='0.5')
                axes[i1, i2].set_xlim(bins[0], bins[-1])
                axes[i1, i2].set_xticks(bins)
                labels = ['']*len(bins)
                labels[::2] = bins[::2]
                axes[i1, i2].set_xticklabels(labels)
                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('# Spikes in burst')
                if i2 == (0):
                    axes[i1, i2].set_ylabel('Frequency')
                axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    adjust_bottom = 0.13 if len(cell_ids) <= 3 else 0.08
    pl.subplots_adjust(left=0.06, bottom=adjust_bottom, top=0.94)
    pl.savefig(os.path.join(save_dir_img, cell_type, 'n_spikes_in_burst.png'))


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/n_spikes_in_burst'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'pyramidal_layer3'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    ISI_burst = 10
    use_AP_max_idxs_domnisoru = True

    n_spikes_in_burst_groups_per_cell = []

    for cell_id in cell_ids:
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
        burst_ISI_bool = ISIs <= ISI_burst
        groups = np.split(burst_ISI_bool, np.where(np.abs(np.diff(burst_ISI_bool)) == 1)[0] + 1)
        burst_groups = []
        for g in groups:
            if True in g:
                burst_groups.append(g)
        n_spikes_in_burst_groups = [len(g) + 1 for g in burst_groups]
        n_spikes_in_burst_groups_per_cell.append(n_spikes_in_burst_groups)

        bins = np.arange(0, 15+1, 1)
        pl.figure()
        pl.hist(n_spikes_in_burst_groups, bins=bins, color='0.5')
        pl.xlabel('# Spikes in burst')
        pl.ylabel('Frequency')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'n_spikes_in_burst.png'))
        #pl.show()

    # plot all cells
    pl.close('all')
    plot_n_spikes_in_burst_all_cells(n_spikes_in_burst_groups_per_cell)
    pl.show()