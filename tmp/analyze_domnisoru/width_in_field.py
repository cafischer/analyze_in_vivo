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
    bin_size = 5  # cm

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    field_width_mean_cells = np.zeros(len(cell_ids))
    field_width_std_cells = np.zeros(len(cell_ids))
    total_in_field_cells = np.zeros(len(cell_ids))

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

        # bin according to position and compute firing rate
        bins = np.arange(0, get_last_bin_edge(cell_id), bin_size)  # use same as matlab's edges

        bins_in_field = get_bins_field_domnisoru(cell_id, save_dir, bins)
        new_field = np.concatenate((np.array([0]), np.where(np.diff(bins_in_field) > 1)[0] + 1, np.array([len(bins_in_field)])))
        bins_in_field = [bins_in_field[i1:i2] for i1, i2 in zip(new_field[:-1], new_field[1:])]

        assert bin_size == 5  # cm
        field_widths = np.array([len(bins)*bin_size for bins in bins_in_field])
        field_width_mean_cells[cell_idx] = np.mean(field_widths)
        field_width_std_cells[cell_idx] = np.std(field_widths)
        total_in_field_cells[cell_idx] = np.sum(field_widths)


    def plot_bar(ax, cell_idx, field_width_mean_cells, field_width_std_cells):
        ax.bar(0, field_width_mean_cells[cell_idx], 0.5, yerr=field_width_std_cells[cell_idx], color='0.5', capsize=2.0)
        ax.set_xlim(-1, 1)
        ax.set_xticks([])

    plot_kwargs = dict(field_width_mean_cells=field_width_mean_cells, field_width_std_cells=field_width_std_cells)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_bar, plot_kwargs,
                            xlabel='', ylabel='Field width (cm)',
                            save_dir_img=os.path.join(save_dir_img, 'field_widths.png'))


    def plot_bar(ax, cell_idx, total_in_field_cells):
        ax.bar(0, total_in_field_cells[cell_idx], 0.5, color='0.5', capsize=2.0)
        ax.set_xlim(-1, 1)
        ax.set_xticks([])

    plot_kwargs = dict(total_in_field_cells=total_in_field_cells)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_bar, plot_kwargs,
                            xlabel='', ylabel='In field area (cm)',
                            save_dir_img=os.path.join(save_dir_img, 'in_field_area.png'))

    pl.show()