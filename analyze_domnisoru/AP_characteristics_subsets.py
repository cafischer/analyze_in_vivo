from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.sta import get_sta_criterion
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.load.load_domnisoru import load_data
pl.style.use('paper_subplots')


if __name__ == '__main__':
    #save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    n_trials = 10
    do_detrend = False
    before_AP = 25
    after_AP = 25
    t_vref = 10.0
    dt = 0.05
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                  + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    save_dir_characteristics = os.path.join(save_dir_characteristics, folder_detrend[do_detrend], folder_name)
    if not os.path.exists(save_dir_characteristics):
        os.makedirs(save_dir_characteristics)

    #
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)
    t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

    # main
    DAP_time_cells = np.zeros(len(cell_ids), dtype=object)
    sem_at_DAP_cells = np.zeros(len(cell_ids), dtype=object)

    # get STA
    param_list = ['Vm_ljpc', 'spiketimes']
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        dt = data['dt']
        t = np.arange(len(v)) * dt
        AP_max_idxs = data['spiketimes']

        len_chunk = int(np.floor(len(v) / n_trials))
        DAP_time_cells[cell_idx] = []
        sem_at_DAP_cells[cell_idx] = []
        for n_trial in range(n_trials):
            print n_trial
            # divide in subsets
            v_trial = v[n_trial * len_chunk:(n_trial+1) * len_chunk]
            t_trial = t[n_trial * len_chunk:(n_trial+1) * len_chunk]
            AP_max_idxs_trial = AP_max_idxs[np.logical_and(t_trial[0] <= t[AP_max_idxs], t[AP_max_idxs] < t_trial[-1])] - n_trial * len_chunk

            (sta_mean, sta_std, sta_mean_good_APs, sta_std_good_APs,
             n_APs_good) = get_sta_criterion(v_trial, dt, AP_max_idxs_trial, do_detrend, before_AP, after_AP,
                                             AP_criterion, t_vref)

            # get AP characteristics from STA
            v_ref = sta_mean[before_AP_idx - to_idx(t_vref, dt)]
            spike_characteristics_dict = get_spike_characteristics_dict()
            spike_characteristics_dict['AP_max_idx'] = before_AP_idx
            spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
            (AP_amp, AP_width, AP_max_idx, DAP_deflection, DAP_amp, DAP_width, DAP_time,
             DAP_max_idx) = get_spike_characteristics(sta_mean, t_AP, ['AP_amp', 'AP_width', 'AP_max_idx', 'DAP_deflection',
                                                                       'DAP_amp', 'DAP_width', 'DAP_time', 'DAP_max_idx'],
                                                      v_ref, check=False, **spike_characteristics_dict)

            if DAP_max_idx is not None:
                sem_at_DAP = sta_std[DAP_max_idx] / np.sqrt(n_APs_good)
                sem_at_DAP_cells[cell_idx].append(sem_at_DAP)

                DAP_time = np.round(DAP_time, 2)
                if DAP_deflection > sem_at_DAP:
                    DAP_time_cells[cell_idx].append(DAP_time)

DAP_cells = get_cell_ids_DAP_cells(new=True)
for cell_id in DAP_cells:
    DAP_times = np.array(DAP_time_cells[cell_ids==cell_id][0])
    print cell_id
    print 'mean: ', np.round(np.mean(DAP_times), 2)
    print 'std: ', np.round(np.std(DAP_times), 2)
    print 'num: ', len(DAP_times)