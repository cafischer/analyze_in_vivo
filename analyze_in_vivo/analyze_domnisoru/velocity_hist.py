from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/velocity_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'vel_100ms']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # parameter
    max_vel = 100  # 200
    bin_width = 0.1  # cm/sec
    bins = np.arange(-1, max_vel+bin_width, bin_width)

    # over cells
    velocity_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity_cells[cell_idx] = data['vel_100ms']

        # print np.min(velocity_cells[cell_idx]), np.max(velocity_cells[cell_idx])
        # pl.figure()
        # pl.plot(t, velocity_cells[cell_idx], 'k')
        # pl.show()


    def plot_vel_hist(ax, cell_idx, velocity_cells, bins, bin_width):
        ax.hist(velocity_cells[cell_idx],
                weights=np.ones(len(velocity_cells[cell_idx])) / float(len(velocity_cells[cell_idx])),
                bins=bins, color='0.5')
        ax.set_xlim(-1, 2)
        ax.set_xticks(range(3))

    plot_kwargs = dict(velocity_cells=velocity_cells, bins=bins, bin_width=bin_width)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_vel_hist, plot_kwargs,
                            xlabel='Vel. (cm/sec)', ylabel='Rel. frequency',
                            save_dir_img=os.path.join(save_dir_img, 'velocity_hist_' + str(bin_width) + '.png'))
    pl.show()