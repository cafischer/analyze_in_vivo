from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells, plot_cumulative_ISI_hist_all_cells_with_bursty
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax, plot_ISI_hist_on_ax_with_kde, get_ISI_hist_peak_and_width
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
import scipy.stats as st
import pandas as pd
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    DAP_cells = get_cell_ids_DAP_cells(new=True)
    param_list = ['Vm_ljpc', 'spiketimes']

    n_trials = 10
    max_ISI = 200  # None if you want to take all ISIs
    burst_ISI = 8  # ms
    bin_width = 1  # ms
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    sigma_smooth = 1  # ms  None for no smoothing
    dt_kde = 0.05
    t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)

    # over cells
    peak_ISI_hist_cells = np.zeros(len(cell_ids), dtype=object)
    width_ISI_hist_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        dt = data['dt']
        t = np.arange(len(v)) * dt
        AP_max_idxs = data['spiketimes']

        len_chunk = int(np.floor(len(v) / n_trials))
        peak_ISI_hist_cells[cell_idx] = []
        width_ISI_hist_cells[cell_idx] = []
        for n_trial in range(n_trials):
            print n_trial
            # divide in subsets
            v_trial = v[n_trial * len_chunk:(n_trial+1) * len_chunk]
            t_trial = t[n_trial * len_chunk:(n_trial+1) * len_chunk]
            AP_max_idxs_trial = AP_max_idxs[np.logical_and(t_trial[0] <= t[AP_max_idxs], t[AP_max_idxs] < t_trial[-1])] - n_trial * len_chunk
            if len(AP_max_idxs_trial) < 5:
                continue

            # ISIs
            ISIs = get_ISIs(AP_max_idxs_trial, t)
            if max_ISI is not None:
                ISIs = ISIs[ISIs <= max_ISI]
            print len(ISIs)
            if len(ISIs) < 5:
                continue

            # compute KDE
            kde = perform_kde(ISIs, sigma_smooth)
            ISI_kde = evaluate_kde(t_kde, kde)

            # compute peak ISI hist
            peak_ISI_hist, width_ISI_hist = get_ISI_hist_peak_and_width(ISI_kde, t_kde)
            peak_ISI_hist_cells[cell_idx].append(peak_ISI_hist)
            width_ISI_hist_cells[cell_idx].append(width_ISI_hist)

            # # plot
            # pl.figure()
            # pl.plot(t_kde, ISI_kde)
            # pl.show()

    for cell_id in DAP_cells:
        peaks_ISI_hist = np.array(peak_ISI_hist_cells[cell_ids == cell_id][0])
        widths_ISI_hist = np.array(width_ISI_hist_cells[cell_ids == cell_id][0])
        print cell_id
        print 'mean: ', np.round(np.mean(peaks_ISI_hist), 2)
        print 'std: ', np.round(np.std(peaks_ISI_hist), 2)
        print 'num: ', len(peaks_ISI_hist)