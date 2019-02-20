from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from grid_cell_stimuli import get_AP_max_idxs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from analyze_in_vivo.analyze_domnisoru.STA.sta import find_all_APs_in_v_trace
from cell_characteristics import to_idx
from grid_cell_stimuli.ISI_hist import get_ISIs


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_field/voltage_before_spike'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'stellate_layer2'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    velocity_threshold = 1  # cm/sec
    ISI_burst = 10
    before_AP = 10
    after_AP = 0

    for cell_id in cell_ids:
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        in_field_len_orig = np.load(os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

        # get phases
        start_in, end_in = get_starts_ends_group_of_ones(in_field_len_orig.astype(int))
        n_fields = len(start_in)

        # get APs
        AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt, interval=2, v_diff_onset_max=5)
        ISIs = get_ISIs(AP_max_idxs, t)
        starts_burst, ends_burst = get_starts_ends_group_of_ones(
            np.concatenate((ISIs <= ISI_burst, np.array([False]))).astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = AP_max_idxs[~np.concatenate((ISIs <= ISI_burst, np.array([False])))]
        AP_max_idxs_single_in_field = AP_max_idxs_single[in_field_len_orig[AP_max_idxs_single]]
        AP_max_idxs_burst_in_field = AP_max_idxs_burst[in_field_len_orig[AP_max_idxs_burst]]

        v_APs_single = find_all_APs_in_v_trace(v, before_AP_idx, after_AP_idx, AP_max_idxs_single_in_field, AP_max_idxs)
        v_APs_burst = find_all_APs_in_v_trace(v, before_AP_idx, after_AP_idx, AP_max_idxs_burst_in_field, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # plots
        pl.figure()
        for v_AP in v_APs_single:
            pl.plot(t_AP, v_AP, 'b', alpha=0.5)
        for v_AP in v_APs_burst:
            pl.plot(t_AP, v_AP, 'r', alpha=0.5)

        pl.figure()
        pl.plot(t_AP, np.mean(v_APs_single, 0), 'b', label='single')
        pl.plot(t_AP, np.mean(v_APs_burst, 0), 'r', label='1st in burst')
        pl.fill_between(t_AP, np.mean(v_APs_single, 0) - np.std(v_APs_single, 0),
                        np.mean(v_APs_single, 0) + np.std(v_APs_single, 0),
                        color='b', alpha=0.5)
        pl.fill_between(t_AP, np.mean(v_APs_burst, 0) - np.std(v_APs_burst, 0),
                        np.mean(v_APs_burst, 0) + np.std(v_APs_burst, 0),
                        color='r', alpha=0.5)
        pl.legend()
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential')
        pl.tight_layout()
        pl.show()