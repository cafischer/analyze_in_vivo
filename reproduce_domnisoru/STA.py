from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.investigate_grid_cell_stimuli import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plots_sta
from grid_cell_stimuli.remove_APs import get_spike_idxs
pl.style.use('paper')


def find_all_APs_in_v_trace(v, before_AP_idx, after_AP_idx, AP_threshold, dt, do_detrend=False, v_detrend=None):
    v_APs = []
    AP_max_idxs = get_spike_idxs(v, AP_threshold, dt)

    for i, AP_max_idx in enumerate(AP_max_idxs):
        if AP_max_idx - before_AP_idx >= 0 and AP_max_idx + after_AP_idx < len(v):  # able to draw window
            v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]

            AP_max_idxs_window = AP_max_idxs[np.logical_and(AP_max_idxs > AP_max_idx - before_AP_idx,
                                                            AP_max_idxs < AP_max_idx + after_AP_idx + 1)]
            AP_max_idxs_window = filter(lambda x: x != AP_max_idx, AP_max_idxs_window)  # remove the AP that should be in the window
            if len(AP_max_idxs_window) == 0:  # check no other APs in the window
                if do_detrend:
                    v_APs.append(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                else:
                    v_APs.append(v_AP)
    return v_APs


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'stellate_layer2'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}
    param_list = ['Vm_ljpc']

    # parameters
    do_detrend = True
    before_AP_sta = 25
    after_AP_sta = 25
    before_AP_stc = 0
    after_AP_stc = 25

    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # detrend
        if do_detrend:
            v_detrend = detrend(v, t, cutoff_freq=5)
        else:
            v_detrend = None

        # plot
        if do_detrend:
            save_dir_cell = os.path.join(save_dir_img, 'detrended', cell_type, cell_id)
        else:
            save_dir_cell = os.path.join(save_dir_img, 'not_detrended', cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        pl.figure()
        if do_detrend:
            pl.plot(t, v_detrend, 'k')
        else:
            pl.plot(t, v, 'k')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'whole_trace.png'))

        # STA
        v_APs = find_all_APs_in_v_trace(v, before_AP_idx_sta, after_AP_idx_sta, AP_thresholds[cell_id], dt,
                                        do_detrend=do_detrend, v_detrend=v_detrend)
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        sta, sta_std = get_sta(v_APs)

        plots_sta(v_APs, t_AP, sta, sta_std, save_dir_cell)
        #pl.show()