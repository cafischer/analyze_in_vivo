from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.sta import get_v_APs
from cell_characteristics.sta_stc import get_sta
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/bootstrap'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_ids = load_cell_ids(save_dir, 'grid_cells')
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    n_bootstraps = 10000
    dt = 0.05  # ms
    do_detrend = False
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend])
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # for paper
    t_vref = 10  # ms
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    time_before_after_AP = (25, 25)  # (before_AP, after_AP)

    # for thesis
    #t_vref = 5  # ms
    #AP_criterions = [{'AP_amp_and_width': (51.8, 0.72)}]
    #AP_criterions = [{'None': None}]
    #time_before_after_AP = [(10, 25)]  # (before_AP, after_AP)

    # main
    sta_mean_cells = np.zeros((len(cell_ids), int(np.round(sum(time_before_after_AP)/dt))+1, n_bootstraps))
    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        dt = data['dt']
        AP_max_idxs = data['spiketimes']

        _, v_APs_good = get_v_APs(v, dt, AP_max_idxs, do_detrend, time_before_after_AP[0], time_before_after_AP[1],
                                  AP_criterion, t_vref)

        # bootstrap APs
        n_APs = len(v_APs_good)

        for i in range(n_bootstraps):
            if n_APs > 10:
                v_APs_bootstrap = v_APs_good[np.random.choice(np.arange(len(v_APs_good)), n_APs, replace=True)]
                sta_mean_cells[cell_idx, :, i], _ = get_sta(v_APs_bootstrap)
            else:
                sta_mean_cells[cell_idx, :, i] = np.nan

    # save
    folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
             + '_before_after_AP_' + str(time_before_after_AP) + '_t_vref_' + str(t_vref)
    if not os.path.exists(os.path.join(save_dir_img, folder)):
        os.makedirs(os.path.join(save_dir_img, folder))
    np.save(os.path.join(save_dir_img, folder, 'sta_mean_bootstrap.npy'), sta_mean_cells)