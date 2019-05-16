from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.sta import get_sta_criterion_all_cells, plot_sta_grid_on_ax
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    dt = 0.05  # ms
    do_detrend = False
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend])
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # AP_criterions = [{'quantile': 20}, {'quantile': 10}, {'AP_amp_and_width': (40, 1)}]
    # time_before_after_AP = [(20, 25), (25, 20), (25, 25), (25, 30), (30, 25), (30, 30)]  # (before_AP, after_AP)

    # for paper
    t_vref = 10  # ms
    AP_criterions = [{'AP_amp_and_width': (40, 1)}]
    time_before_after_AP = [(25, 25)]  # (before_AP, after_AP)

    # for thesis
    #t_vref = 5  # ms
    #AP_criterions = [{'AP_amp_and_width': (51.8, 0.72)}]
    #AP_criterions = [{'None': None}]
    #time_before_after_AP = [(10, 25)]  # (before_AP, after_AP)

    # main
    for AP_criterion in AP_criterions:
        for (before_AP, after_AP) in time_before_after_AP:
            print AP_criterion, (before_AP, after_AP)
            (sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
             sta_std_good_APs_cells, _) = get_sta_criterion_all_cells(do_detrend, before_AP, after_AP,
                                                                      AP_criterion, t_vref, cell_ids, save_dir)

            # save
            folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                     + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
            if not os.path.exists(os.path.join(save_dir_img, folder)):
                os.makedirs(os.path.join(save_dir_img, folder))
            np.save(os.path.join(save_dir_img, folder, 'sta_mean.npy'), sta_mean_good_APs_cells)

            t_AP = np.arange(-before_AP, after_AP + dt, dt)
            plot_kwargs = dict(t_AP=t_AP,
                               sta_mean_cells=sta_mean_cells,
                               sta_std_cells=sta_std_cells,
                               sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                               sta_std_good_APs_cells=sta_std_good_APs_cells,
                               before_AP=before_AP,
                               after_AP=after_AP,
                               ylims=(-75, -45)
                               )

            fig_title = 'Criterion: ' + AP_criterion.keys()[0].replace('_', ' ') + ' ' + str(AP_criterion.values()[0]) \
                        + '  Time range (before AP, after AP): ' + str((before_AP, after_AP))
            plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid_on_ax, plot_kwargs,
                                         xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                         fig_title=fig_title,
                                         save_dir_img=os.path.join(save_dir_img, folder, 'sta.png'))

            #pl.show()