from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_APs
from grid_cell_stimuli import find_all_AP_traces
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.sta import plot_sta
pl.style.use('paper_subplots')


def plot_v_APs(ax, cell_idx, t_AP, v_APs_cells):
    for v_AP in v_APs_cells[cell_idx]:
        ax.plot(t_AP, v_AP)
    ax.set_ylim(-90, 30)
    ax.set_xticks([-10, 0, 10, 20])


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/sorted'
    save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    do_detrend = False
    kind = 'all'
    before_AP = 10
    after_AP = 25
    n_plot = 10
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms', 'fY_cm', 'fvel_100ms']
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], kind, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    v_APs_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

        # get APs
        AP_max_idxs = data['spiketimes']
        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt - before_AP

        # sort according to delta_DAP
        v_APs_cut = v_APs[:, before_AP_idx:before_AP_idx+to_idx(10, dt)]
        global_min_idxs = np.argmin(v_APs_cut, axis=1)
        global_mins = np.min(v_APs_cut, axis=1)
        global_maxs_after_min = np.array([np.max(v_AP[global_min_idxs[i]:]) for i, v_AP in enumerate(v_APs_cut)])
        delta_DAPs = global_maxs_after_min - global_mins
        sort_idxs = np.argsort(delta_DAPs)

        v_APs = v_APs[sort_idxs, :][::-1]
        v_APs_cells[cell_idx] = v_APs[:n_plot, :]
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs[:n_plot, :])

        # pl.figure()
        # for v_AP in v_APs[:n_plot, :]:
        #     pl.plot(t_AP, v_AP)
        #
        # pl.figure()
        # for v_AP in v_APs[-n_plot:, :]:
        #     pl.plot(t_AP, v_AP)
        # pl.show()

    # plot STA
    plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_cells, sta_std_cells=sta_std_cells)
    plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                            save_dir_img=os.path.join(save_dir_img, 'APs_'+str(before_AP)+'_'+str(after_AP)+'.png'))

    # plot APs
    plot_kwargs = dict(t_AP=t_AP, v_APs_cells=v_APs_cells)
    plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_v_APs, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                            save_dir_img=os.path.join(save_dir_img, 'sta_'+str(before_AP)+'_'+str(after_AP)+'.png'))
    pl.show()