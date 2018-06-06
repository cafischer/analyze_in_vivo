from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/entropy'
    save_dir_firing_rate = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    inv_entropy = np.zeros(len(cell_ids))
    position_cells = []
    firing_rate_cells = []
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        firing_rate = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'firing_rate.npy'))
        position = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'position.npy'))
        firing_rate_cells.append(firing_rate)
        position_cells.append(position)

        # entropy
        position = position[~np.isnan(firing_rate)]
        firing_rate = firing_rate[~np.isnan(firing_rate)]
        prob_position = firing_rate / np.sum(firing_rate)
        summands = prob_position * (np.log(prob_position) / np.log(2))
        summands[np.isnan(summands)] = 0  # by definition: if prob=0, then summand = 0
        entropy_cell = -np.sum(summands)

        prob_position_uniform = np.ones(len(position)) / (len(position))
        summands = prob_position_uniform * (np.log(prob_position_uniform) / np.log(2))
        entropy_uniform = -np.sum(summands)
        inv_entropy[cell_idx] = 1 - entropy_cell / entropy_uniform

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
                if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                    ax1.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax1.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.plot(position_cells[cell_idx], firing_rate_cells[cell_idx], 'k')
                ax1.annotate('Spatial inf.: %.2f' % inv_entropy[cell_idx],
                             xy=(0.1, 0.9), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                             bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))
                if i >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('Position (cm)', fontsize=10)
                if i % n_columns == 0:
                    ax1.set_ylabel('Firing rate (Hz)')
                fig.add_subplot(ax1)
                cell_idx += 1
        pl.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.07)
        pl.savefig(os.path.join(save_dir_img, 'entropy.png'))
        pl.show()