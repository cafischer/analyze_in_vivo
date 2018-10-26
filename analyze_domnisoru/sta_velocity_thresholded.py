from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import find_all_AP_traces
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import threshold_by_velocity
pl.style.use('paper')


def get_AP_max_idxs_above_and_under_velocity_threshold(AP_max_idxs, velocity, velocity_threshold):
    spike_train = init_nan(len(velocity))
    spike_train[AP_max_idxs] = AP_max_idxs
    spike_train = threshold_by_velocity([spike_train], velocity, velocity_threshold)[0][0]
    AP_max_idxs_above = spike_train[~np.isnan(spike_train)].astype(int)
    AP_max_idxs_under = np.setxor1d(AP_max_idxs, AP_max_idxs_above).astype(int)
    return AP_max_idxs_above, AP_max_idxs_under


def plot_sta(ax, cell_idx, subplot_idx, t_AP, sta_mean_above_cells, sta_std_above_cells,
             sta_mean_under_cells, sta_std_under_cells, xlims=(None, None), ylims=(None, None)):
    if subplot_idx == 0:
        ax.plot(t_AP, sta_mean_above_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_above_cells[cell_idx] - sta_std_above_cells[cell_idx],
                        sta_mean_above_cells[cell_idx] + sta_std_above_cells[cell_idx], color='k', alpha=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.annotate('$\geq$ vel. thresh.', xy=(xlims[0], ylims[1]), textcoords='data',
                    horizontalalignment='left', verticalalignment='top', fontsize=9)
    if subplot_idx == 1:
        ax.plot(t_AP, sta_mean_under_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_under_cells[cell_idx] - sta_std_under_cells[cell_idx],
                        sta_mean_under_cells[cell_idx] + sta_std_under_cells[cell_idx], color='k', alpha=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.annotate('$<$ vel. thresh.', xy=(xlims[0], ylims[1]), textcoords='data',
                    horizontalalignment='left', verticalalignment='top', fontsize=9)


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/velocity_thresholding'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    velocity_threshold = 1  # cm/sec
    before_AP_sta = 25
    after_AP_sta = 25
    bins_v = np.arange(-90, 40+0.5, 0.5)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms', 'fY_cm', 'fvel_100ms']
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_above_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_above_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_under_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_under_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)
        AP_max_idxs = data['spiketimes']

        # divide into under and above velocity threshold
        AP_max_idxs_above, AP_max_idxs_under = get_AP_max_idxs_above_and_under_velocity_threshold(AP_max_idxs,
                                                                                                  velocity,
                                                                                                  velocity_threshold)

        v_APs_above = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_above, AP_max_idxs)
        v_APs_under = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_under, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt - before_AP_sta

        # get STA
        if v_APs_above is None:
            sta_mean_above_cells[cell_idx] = init_nan(after_AP_idx_sta + before_AP_idx_sta + 1)
            sta_std_above_cells[cell_idx] = init_nan(after_AP_idx_sta + before_AP_idx_sta + 1)
        else:
            sta_mean_above_cells[cell_idx], sta_std_above_cells[cell_idx] = get_sta(v_APs_above)
        if v_APs_under is None:
            sta_mean_under_cells[cell_idx] = init_nan(after_AP_idx_sta + before_AP_idx_sta + 1)
            sta_std_under_cells[cell_idx] = init_nan(after_AP_idx_sta + before_AP_idx_sta + 1)
        else:
            sta_mean_under_cells[cell_idx], sta_std_under_cells[cell_idx] = get_sta(v_APs_under)

    # plots
    plot_kwargs = dict(t_AP=t_AP, sta_mean_above_cells=sta_mean_above_cells, sta_std_above_cells=sta_std_above_cells,
                       sta_mean_under_cells=sta_mean_under_cells, sta_std_under_cells=sta_std_under_cells,
                       xlims=(-before_AP_sta, after_AP_sta), ylims=(-85, 25))

    plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                     xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                     save_dir_img=os.path.join(save_dir_img, 'sta.png'))

    plot_kwargs = dict(t_AP=t_AP, sta_mean_above_cells=sta_mean_above_cells, sta_std_above_cells=sta_std_above_cells,
                       sta_mean_under_cells=sta_mean_under_cells, sta_std_under_cells=sta_std_under_cells,
                       xlims=(-3, after_AP_sta), ylims=(-85, -40))

    plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                     xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                     save_dir_img=os.path.join(save_dir_img, 'sta_zoom.png'))
    pl.show()