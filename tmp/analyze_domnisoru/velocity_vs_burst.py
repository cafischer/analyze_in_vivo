from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_groups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']
    use_AP_max_idxs_domnisoru = True
    ISI_burst = 8  # ms

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs = np.zeros(len(cell_ids), dtype=object)
    velocity_at_APs = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']

        # compute median preceding silence
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']

        ISIs[cell_idx] = get_ISIs(AP_max_idxs, t)

        # burst_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        # starts_burst, ends_burst = get_starts_ends_group_of_ones(burst_ISI_indicator.astype(int))
        # AP_max_idxs_burst = AP_max_idxs[starts_burst]

        velocity_at_APs[cell_idx] = velocity[AP_max_idxs]

    # save and plot
    if cell_type == 'grid_cells':
        def plot_velocity_vs_ISI(ax, cell_idx, velocity_at_APs, ISIs):
            ax.plot(velocity_at_APs[cell_idx][:-1], ISIs[cell_idx], 'ok', markersize=4, alpha=0.5)
            ax.set_ylim(0, 200)

        plot_kwargs = dict(velocity_at_APs=velocity_at_APs, ISIs=ISIs)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_velocity_vs_ISI, plot_kwargs,
                                xlabel='Velocity \n(cm/sec)', ylabel='ISI (ms)',
                                save_dir_img=os.path.join(save_dir_img, 'velocity_vs_ISI.png'))

    pl.show()