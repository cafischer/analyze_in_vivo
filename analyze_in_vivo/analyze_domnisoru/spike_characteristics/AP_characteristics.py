from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.sta import get_sta_criterion_all_cells
from cell_characteristics.analyze_APs import get_spike_characteristics
from analyze_in_vivo.analyze_domnisoru.spike_characteristics import get_spike_characteristics_dict
from cell_fitting.util import init_nan
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_characteristics = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    #save_dir_characteristics = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/'
    #save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    do_detrend = False
    # before_AP = 25
    # after_AP = 25
    # t_vref = 10
    # dt = 0.05
    # AP_criterion = {'AP_amp_and_width': (40, 1)}
    before_AP = 10
    after_AP = 25
    t_vref = 5
    dt = 0.05
    AP_criterion = {'None': None}

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
    AP_amp_cells = np.zeros(len(cell_ids))
    AP_abs_amp_cells = np.zeros(len(cell_ids))
    AP_width_cells = np.zeros(len(cell_ids))
    DAP_deflection_cells = np.zeros(len(cell_ids))
    DAP_amp_cells = np.zeros(len(cell_ids))
    DAP_width_cells = np.zeros(len(cell_ids))
    DAP_time_cells = np.zeros(len(cell_ids))
    sem_at_DAP_cells = init_nan(len(cell_ids))

    # get STA
    (sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
     sta_std_good_APs_cells, n_APs_good_cells) = get_sta_criterion_all_cells(do_detrend, before_AP, after_AP,
                                                                             AP_criterion, t_vref, cell_ids, save_dir)

    for cell_idx, cell_id in enumerate(cell_ids):
        sta_mean, sta_std = sta_mean_good_APs_cells[cell_idx], sta_std_good_APs_cells[cell_idx]

        # get AP characteristics from STA
        v_rest = sta_mean[before_AP_idx - to_idx(t_vref, dt)]
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
        AP_amp_cells[cell_idx], AP_width_cells[cell_idx], AP_max_idx, \
        DAP_deflection_cells[cell_idx], DAP_amp_cells[cell_idx], \
        DAP_width_cells[cell_idx], DAP_time_cells[cell_idx], \
        DAP_max_idx = get_spike_characteristics(sta_mean, t_AP, ['AP_amp', 'AP_width', 'AP_max_idx', 'DAP_deflection',
                                                                 'DAP_amp', 'DAP_width', 'DAP_time', 'DAP_max_idx'],
                                                v_rest, check=False, **spike_characteristics_dict)
        AP_abs_amp_cells[cell_idx] = sta_mean[AP_max_idx]

        # test if DAP deflection greater than
        if DAP_max_idx is not None:
            sem_at_DAP_cells[cell_idx] = sta_std[DAP_max_idx] / np.sqrt(n_APs_good_cells[cell_idx])

    not_nan = ~np.isnan(sem_at_DAP_cells)
    cell_ids = np.array(cell_ids)
    print 'cells with DAP defl. > sem', cell_ids[not_nan][DAP_deflection_cells[not_nan] > sem_at_DAP_cells[not_nan]]
    print 'cells without DAP defl. > sem', cell_ids[not_nan][DAP_deflection_cells[not_nan] < sem_at_DAP_cells[not_nan]]
    print 'DAP defl.: ', ['%.2f' % DAP_d for DAP_d in DAP_deflection_cells[not_nan]]
    print 'sem: ', ['%.2f' % sem for sem in sem_at_DAP_cells[not_nan]]

    np.save(os.path.join(save_dir_characteristics, 'DAP_deflection.npy'), DAP_deflection_cells)
    np.save(os.path.join(save_dir_characteristics, 'DAP_amp.npy'), DAP_amp_cells)
    np.save(os.path.join(save_dir_characteristics, 'DAP_width.npy'), DAP_width_cells)
    np.save(os.path.join(save_dir_characteristics, 'DAP_time.npy'), DAP_time_cells)
    np.save(os.path.join(save_dir_characteristics, 'AP_width.npy'), AP_width_cells)
    np.save(os.path.join(save_dir_characteristics, 'AP_amp.npy'), AP_amp_cells)
    np.save(os.path.join(save_dir_characteristics, 'AP_abs_amp.npy'), AP_abs_amp_cells)

    # t_vref = 5
    # cells with DAP defl. > sem['s67_0000' 's73_0004' 's79_0003' 's104_0007' 's109_0002' 's110_0002'  's119_0004']
    # DAP defl.: [0.99 1.10 0.58 1.65 1.62 2.63 0.67]
    # sem:       [0.26 0.40 0.17 0.26 0.10 0.29 0.12]

    # t_vref = 10
    # cells with DAP defl. > sem ['s67_0000' 's73_0004' 's79_0003' 's104_0007' 's109_0002' 's110_0002'  's119_0004']
    # DAP defl.:  ['1.02', '0.86', '0.58', '1.65', '1.62', '2.63', '0.67']
    # sem:  ['0.26', '0.40', '0.17', '0.26', '0.10', '0.29', '0.12']