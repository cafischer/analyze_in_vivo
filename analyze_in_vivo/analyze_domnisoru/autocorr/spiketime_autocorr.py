from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.autocorr import *
from grid_cell_stimuli.ISI_hist import get_ISIs
import pandas as pd
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, \
    get_cell_ids_DAP_cells
pl.style.use('paper')


if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_ids = load_cell_ids(save_dir, 'non_grid_cells')
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    bin_width = 1  # ms
    max_lag = 1000
    normalization = 'sum'  # 'sum
    save_dir_img = os.path.join(save_dir_img)

    folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_normalization_' + str(normalization)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    autocorr_cells = np.zeros((len(cell_ids), int(2 * max_lag / bin_width + 1)))

    median_velocity = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # get ISIs
        ISIs = get_ISIs(AP_max_idxs, t)

        # get autocorrelation
        autocorr_cells[cell_idx, :], t_autocorr, bins = get_autocorrelation_by_ISIs(ISIs, max_lag=max_lag,
                                                                                    bin_width=bin_width,
                                                                                    normalization=normalization)

        # pl.figure()
        # pl.bar(bins[:-1], autocorr_cells[cell_idx, :], bin_width, color='r', align='center', alpha=0.5)
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # pl.show()

    # save autocorrelation
    np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_cells)

    # plots
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
                       max_lag=max_lag)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                            save_dir_img=os.path.join(save_dir_img, 'autocorr.png'))

    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
    #                         xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
    #                         wspace=0.18,
    #                         save_dir_img=os.path.join(save_dir_img2, 'auto_corr_' + str(max_lag) + '.png'))