from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.sta import get_in_or_out_field_AP_max_idxs
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    kind = 'in_field'
    use_AP_max_idxs_domnisoru = True
    before_AP_sta = 25
    after_AP_sta = 25

    save_dir_characteristics = os.path.join(save_dir_characteristics, kind, cell_type)
    if not os.path.exists(save_dir_characteristics):
        os.makedirs(save_dir_characteristics)

    # main
    AP_amp_per_cell = np.zeros(len(cell_ids))
    AP_width_per_cell = np.zeros(len(cell_ids))
    DAP_deflection_per_cell = np.zeros(len(cell_ids))
    DAP_width_per_cell = np.zeros(len(cell_ids))
    DAP_time_per_cell = np.zeros(len(cell_ids))
    sem_at_DAP = init_nan(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        before_AP_idx = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        AP_max_idxs_selected = get_in_or_out_field_AP_max_idxs(kind, AP_max_idxs, velocity, cell_id, save_dir)

        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx_sta, AP_max_idxs_selected, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx + 1) * dt
        if v_APs is None:
            continue

        # get AP characteristics from STA
        sta_mean, sta_std = get_sta(v_APs)
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1, dt)
        AP_amp_per_cell[cell_idx], AP_width_per_cell[cell_idx], DAP_deflection_per_cell[cell_idx], \
        DAP_width_per_cell[cell_idx], DAP_time_per_cell[cell_idx], DAP_max_idx = get_spike_characteristics(sta_mean, t_AP,
                                                                                              ['AP_amp', 'AP_width', 'DAP_deflection',
                                                                                               'DAP_width', 'DAP_time', 'DAP_max_idx'],
                                                                                              sta_mean[0], check=False,
                                                                                              **spike_characteristics_dict)

        # test if DAP deflection greater than
        if DAP_max_idx is not None:
            sem_at_DAP[cell_idx] = sta_std[DAP_max_idx] / np.sqrt(len(v_APs))

    not_nan = ~np.isnan(sem_at_DAP)
    cell_ids = np.array(cell_ids)
    print cell_ids[not_nan][DAP_deflection_per_cell[not_nan] > sem_at_DAP[not_nan]]
    print cell_ids[not_nan][DAP_deflection_per_cell[not_nan] < sem_at_DAP[not_nan]]

    np.save(os.path.join(save_dir_characteristics, 'DAP_deflection.npy'), DAP_deflection_per_cell)
    np.save(os.path.join(save_dir_characteristics, 'DAP_width.npy'), DAP_width_per_cell)
    np.save(os.path.join(save_dir_characteristics, 'DAP_time.npy'), DAP_time_per_cell)
    np.save(os.path.join(save_dir_characteristics, 'AP_width.npy'), AP_width_per_cell)
    np.save(os.path.join(save_dir_characteristics, 'AP_amp.npy'), AP_amp_per_cell)