from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['vel_100ms', 'fvel_100ms']

    max_vel = 60  # cm/sec
    bin_width = 1  # cm/sec
    bins = np.arange(-1, max_vel+bin_width, bin_width)

    # over cells
    median_velocity = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        velocity = data['fvel_100ms']

        median_velocity[cell_idx] = np.median(velocity)

    # plot
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if id in cell_ids_bursty else False for id in cell_ids])

    np.save(os.path.join('/home/cf', 'median_velocity.npy'), median_velocity)
    np.save(os.path.join('/home/cf', 'burst_label.npy'), burst_label)

    pl.figure()
    pl.hist(median_velocity[burst_label], bins, color='r', alpha=0.5)
    pl.hist(median_velocity[~burst_label], bins, color='b', alpha=0.5)
    pl.xlabel('Median velocity')
    pl.ylabel('Number of cells')
    pl.show()