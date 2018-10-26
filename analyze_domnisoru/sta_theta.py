from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, load_field_indices, get_celltype_dict
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid
from analyze_in_vivo.analyze_domnisoru.sta_velocity_thresholded import plot_sta
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/theta'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)  #[15:18]

    # parameters
    use_AP_max_idxs_domnisoru = True
    kind = 'all'
    before_AP = 10
    after_AP = 25
    bins_v = np.arange(-75, 40+0.5, 0.5)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'fVm']
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, kind, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    DAP_deflections = {}
    sta_mean_theta_high_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_theta_high_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_theta_low_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_theta_low_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        theta = data['fVm']
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        theta_APs = find_all_AP_traces(theta, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt - before_AP

        # divide by high/low theta amplitude
        theta_median = np.median(np.abs(theta))
        print theta_median
        APs_high_theta = np.zeros(np.shape(v_APs)[0], dtype=bool)
        APs_low_theta = np.zeros(np.shape(v_APs)[0], dtype=bool)
        for i, theta_AP in enumerate(theta_APs):
            if np.mean(np.abs(theta_AP)) >= 3.0:
                APs_high_theta[i] = True
            elif np.mean(np.abs(theta_AP)) <= 1.0:
                APs_low_theta[i] = True

        # STA
        sta_mean_theta_high_cells[cell_idx], sta_std_theta_high_cells[cell_idx] = get_sta(v_APs[APs_high_theta, :])
        sta_mean_theta_low_cells[cell_idx], sta_std_theta_low_cells[cell_idx] = get_sta(v_APs[APs_low_theta, :])

    # save
    save_dir_cell = os.path.join(save_dir_img, cell_id)
    if not os.path.exists(save_dir_cell):
        os.makedirs(save_dir_cell)

    plot_kwargs = dict(t_AP=t_AP, sta_mean_above_cells=sta_mean_theta_high_cells,
                       sta_std_above_cells=sta_std_theta_high_cells,
                       sta_mean_under_cells=sta_mean_theta_low_cells,
                       sta_std_under_cells=sta_std_theta_low_cells,
                       xlims=(-before_AP, after_AP), ylims=(-70, -40))

    plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                 xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                 save_dir_img=os.path.join(save_dir_img, 'sta.png'))

    pl.show()