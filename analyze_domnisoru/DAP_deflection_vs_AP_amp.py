from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, get_sta_median
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/DAP_deflection_vs_AP_amp'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
    before_AP_sta = 25
    after_AP_sta = 25
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    #
    DAP_deflection_per_cell = np.zeros(len(cell_ids))
    AP_max_per_cell = np.zeros(len(cell_ids))

    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        v_APs = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        if v_APs is None:
            continue

        # DAP_deflection and AP max from STA
        sta_mean, sta_std = get_sta(v_APs)
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx_sta
        spike_characteristics_dict['AP_onset'] = 0
        DAP_deflection = get_spike_characteristics(sta_mean, t_AP, ['DAP_deflection'], sta_mean[0],
                                                             check=False, **spike_characteristics_dict)[0]
        DAP_deflection_per_cell[i] = DAP_deflection if DAP_deflection is not None else 0
        AP_max_per_cell[i] = np.max(sta_mean) - sta_mean[before_AP_idx_sta-to_idx(1, dt)]

    pl.figure()
    pl.plot(AP_max_per_cell, DAP_deflection_per_cell, 'ok')
    pl.ylabel('DAP deflection (mV)')
    pl.xlabel('AP amplitude (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_amp.png'))
    pl.show()