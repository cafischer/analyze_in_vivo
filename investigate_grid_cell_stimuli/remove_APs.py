from __future__ import division
import numpy as np
import os
import json
from load import load_field_crossings, get_stellate_info
from grid_cell_stimuli.remove_APs import remove_APs, plot_v_APs_removed


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/APs_removed'
    save_dir_data = '../results/' + folder + '/data'

    # parameters
    t_before = 3
    t_after = 6

    # over all field crossings
    for file_name in os.listdir(save_dir_data):

        # load
        v = np.load(os.path.join(save_dir_data, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        dt = t[1] - t[0]

        # remove APs
        AP_threshold = np.max(v) - np.abs((np.min(v) - np.max(v)) / 3)
        v_APs_removed = remove_APs(v, t, AP_threshold, t_before, t_after)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        np.save(os.path.join(save_dir_cell_field_crossing, 'v.npy'), v_APs_removed)
        np.save(os.path.join(save_dir_cell_field_crossing, 't.npy'), t)

        params = {'AP_threshold': AP_threshold, 't_before': t_before, 't_after': t_after}
        with open(os.path.join(save_dir_cell_field_crossing, 'params'), 'w') as f:
            json.dump(params, f)

        plot_v_APs_removed(v_APs_removed, v, t, save_dir_cell_field_crossing)