from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from analyze_in_vivo.analyze_domnisoru.sta import plot_sta_grid_on_ax
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid, \
    plot_with_markers
pl.style.use('paper_subplots')


def get_sta_criterion(do_detrend, before_AP, after_AP, AP_criterion):
    sta_mean_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        AP_max_idxs = data['spiketimes']

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt
        if v_APs is None:
            continue

        # get AP amp. and width
        AP_amps = np.zeros(len(v_APs))
        AP_widths = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
            AP_amps[i], AP_widths[i] = get_spike_characteristics(v_AP, t_AP, ['AP_amp', 'AP_width'],
                                                                 v_rest=v_AP[before_AP_idx - to_idx(t_v_ref, dt)],
                                                                 AP_max_idx=before_AP_idx,
                                                                 AP_onset=before_AP_idx - to_idx(1.0, dt),
                                                                 std_idx_times=(0, 1), check=False,
                                                                 **spike_characteristics_dict)

        # select APs
        if AP_criterion.keys()[0] == 'quantile':
            AP_amp_thresh = np.nanpercentile(AP_amps, 100 - AP_criterion.values()[0])
            AP_width_thresh = np.nanpercentile(AP_widths, AP_criterion.values()[0])
        elif AP_criterion.keys()[0] == 'AP_amp_and_width':
            AP_amp_thresh = AP_criterion.values()[0][0]
            AP_width_thresh = AP_criterion.values()[0][1]
        else:
            raise ValueError('AP criterion does not exist!')

        good_APs = np.logical_and(AP_amps >= AP_amp_thresh, AP_widths <= AP_width_thresh)
        v_APs_good = v_APs[good_APs]

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        if len(v_APs_good) > 5:
            sta_mean_good_APs_cells[cell_idx], sta_std_good_APs_cells[cell_idx] = get_sta(v_APs_good)
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))

    return sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells, sta_std_good_APs_cells


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    t_v_ref = 10  # ms
    dt = 0.05  # ms
    do_detrend = False
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend])
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    AP_criterions = [{'quantile': 20}, {'AP_amp_and_width': (30, 1)}, {'AP_amp_and_width': (40, 1)}]
    time_before_after_AP = [(10, 15), (15, 15), (15, 20), (20, 20)]  # (before_AP, after_AP)

    # main
    for AP_criterion in AP_criterions:
        for (before_AP, after_AP) in time_before_after_AP:
            (sta_mean_cells, sta_std_cells,
             sta_mean_good_APs_cells, sta_std_good_APs_cells) = get_sta_criterion(do_detrend, before_AP, after_AP,
                                                                                  AP_criterion)

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
            fig_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                       + '_before_after_AP_' + str((before_AP, after_AP)) + '.png'

            plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid_on_ax, plot_kwargs,
                                         xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                         fig_title=fig_title,
                                         save_dir_img=os.path.join(save_dir_img, fig_name))

            #pl.show()