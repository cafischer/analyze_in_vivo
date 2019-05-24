from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    len_recording = np.zeros(len(cell_ids))
    firing_rate = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # compute firing rate
        len_recording[cell_idx] = t[-1]
        firing_rate[cell_idx] = len(AP_max_idxs) / (len_recording[cell_idx] / 1000.)  # Hz

    # save
    np.save(os.path.join(save_dir_img, 'firing_rate.npy'), firing_rate)