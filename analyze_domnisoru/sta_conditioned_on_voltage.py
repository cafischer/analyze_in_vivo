from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_sta_median, plot_APs
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_fitting.util import init_nan
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
    use_mean_or_std = 'std'
    do_detrend = False
    in_field = False
    out_field = False
    before_AP_sta = 25
    after_AP_sta = 25
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_greater_cells = []
    sta_std_greater_cells = []
    sta_mean_lower_cells = []
    sta_std_lower_cells = []
    DAP_deflection_greater_per_cell = np.zeros(len(cell_ids))
    DAP_width_greater_per_cell = np.zeros(len(cell_ids))
    DAP_time_greater_per_cell = np.zeros(len(cell_ids))
    DAP_deflection_lower_per_cell = np.zeros(len(cell_ids))
    DAP_width_lower_per_cell = np.zeros(len(cell_ids))
    DAP_time_lower_per_cell = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        if in_field:
            in_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[in_field_len_orig[AP_max_idxs]]
        elif out_field:
            out_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'out_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[out_field_len_orig[AP_max_idxs]]
        else:
            AP_max_idxs_selected = AP_max_idxs

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_selected, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt

        # mean voltage before AP
        v_condition = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            if use_mean_or_std == 'mean':
                v_condition[i] = np.mean(v_AP[:before_AP_idx_sta - to_idx(1, dt)])
            elif use_mean_or_std == 'std':
                v_condition[i] = np.std(v_AP[:before_AP_idx_sta - to_idx(1, dt)])
        half = np.percentile(v_condition, 50)
        v_APs_greater = v_APs[v_condition > half]
        v_APs_lower = v_APs[v_condition <= half]

        # STA
        sta_mean_greater, sta_std_greater = get_sta(v_APs_greater)
        sta_mean_lower, sta_std_lower = get_sta(v_APs_lower)
        sta_mean_greater_cells.append(sta_mean_greater)
        sta_std_greater_cells.append(sta_std_greater)
        sta_mean_lower_cells.append(sta_mean_lower)
        sta_std_lower_cells.append(sta_std_lower)

        # plot_APs(v_APs_lower, t_AP, None)
        # plot_APs(v_APs_greater, t_AP, None)
        # pl.show()

        # get DAP characteristics
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx_sta
        spike_characteristics_dict['AP_onset'] = before_AP_idx_sta - to_idx(1, dt)
        spike_characteristics_dict['order_fAHP_min'] = 0.2
        spike_characteristics_dict['fAHP_interval'] = 3.0
        spike_characteristics_dict['DAP_interval'] = 5.0
        DAP_time_greater_per_cell[cell_idx], DAP_deflection_greater_per_cell[cell_idx], \
        DAP_width_greater_per_cell[cell_idx] = get_spike_characteristics(sta_mean_greater, t_AP,
                                                                        ['fAHP2DAP_time', 'DAP_deflection', 'DAP_width'],
                                                                        sta_mean_greater[0],
                                                                        check=False, **spike_characteristics_dict)
        DAP_time_lower_per_cell[cell_idx], DAP_deflection_lower_per_cell[cell_idx], \
        DAP_width_lower_per_cell[cell_idx] = get_spike_characteristics(sta_mean_lower, t_AP,
                                                                        ['fAHP2DAP_time', 'DAP_deflection', 'DAP_width'],
                                                                        sta_mean_lower[0],
                                                                        check=False, **spike_characteristics_dict)
    # plots

    if use_mean_or_std == 'mean':
        labels = ['$\mu$ > 50th percentile', '$\mu$ <= 50th percentile']
    elif use_mean_or_std == 'std':
        labels = ['$\sigma$ > 50th percentile', '$\sigma$ <= 50th percentile']

    #
    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                    offset = 10
                    axes[i1, i2].plot(t_AP, sta_mean_greater_cells[cell_idx] + offset, 'r', label=labels[0])
                    axes[i1, i2].fill_between(t_AP,
                                              sta_mean_greater_cells[cell_idx] - sta_std_greater_cells[cell_idx] + offset,
                                              sta_mean_greater_cells[cell_idx] + sta_std_greater_cells[cell_idx] + offset,
                                              color='r', alpha=0.5)
                    axes[i1, i2].plot(t_AP, sta_mean_lower_cells[cell_idx], 'b', label=labels[1])
                    axes[i1, i2].fill_between(t_AP, sta_mean_lower_cells[cell_idx] - sta_std_lower_cells[cell_idx],
                                              sta_mean_lower_cells[cell_idx] + sta_std_lower_cells[cell_idx],
                                              color='b', alpha=0.5)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('Time (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Mem. Pot. (mV)')
                    if i1 == 0 and i2 == (n_columns - 1):
                        axes[i1, i2].legend(fontsize=10)
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        if use_mean_or_std == 'mean':
            pl.savefig(os.path.join(save_dir_img, 'sta_conditioned_on_mean_v.png'))
        elif use_mean_or_std == 'std':
            pl.savefig(os.path.join(save_dir_img, 'sta_conditioned_on_std_v.png'))

        #
        pl.figure()
        lim_max = max(np.nanmax(DAP_deflection_lower_per_cell), np.nanmax(DAP_deflection_greater_per_cell)) + 0.1
        pl.plot(np.arange(0, lim_max + 0.1, 0.1), np.arange(0, lim_max + 0.1, 0.1), color='0.5', linestyle='--')
        pl.plot(DAP_deflection_lower_per_cell, DAP_deflection_greater_per_cell, 'ok')
        for cell_idx in range(len(cell_ids)):
            pl.annotate(cell_ids[cell_idx], xy=(DAP_deflection_lower_per_cell[cell_idx] + 0.05,
                                                DAP_deflection_greater_per_cell[cell_idx] + 0.05))
        pl.title('DAP deflection (mV)')
        pl.xlim(0, lim_max)
        pl.ylim(0, lim_max)
        pl.xlabel(labels[1])
        pl.ylabel(labels[0])
        pl.tight_layout()
        if use_mean_or_std == 'mean':
            pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_conditioned_on_mean_v.png'))
        elif use_mean_or_std == 'std':
            pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_conditioned_on_std_v.png'))

        pl.figure()
        lim_min = min(np.nanmin(DAP_time_lower_per_cell), np.nanmin(DAP_time_greater_per_cell)) + 0.1
        lim_max = max(np.nanmax(DAP_time_lower_per_cell), np.nanmax(DAP_time_greater_per_cell)) + 0.1
        pl.plot(np.arange(lim_min - 0.1, lim_max + 0.1, 0.1), np.arange(lim_min - 0.1, lim_max + 0.1, 0.1),
                color='0.5', linestyle='--')
        pl.plot(DAP_time_lower_per_cell, DAP_time_greater_per_cell, 'ok')
        for cell_idx in range(len(cell_ids)):
            pl.annotate(cell_ids[cell_idx], xy=(DAP_time_lower_per_cell[cell_idx] + 0.05,
                                                DAP_time_greater_per_cell[cell_idx] + 0.05))
        pl.title('Time fAHP - DAP')
        pl.xlabel(labels[1])
        pl.ylabel(labels[0])
        pl.tight_layout()
        if use_mean_or_std == 'mean':
            pl.savefig(os.path.join(save_dir_img, 'DAP_time_conditioned_on_mean_v.png'))
        elif use_mean_or_std == 'std':
            pl.savefig(os.path.join(save_dir_img, 'DAP_time_conditioned_on_std_v.png'))

        pl.figure()
        lim_min = min(np.nanmin(DAP_width_lower_per_cell), np.nanmin(DAP_width_greater_per_cell)) + 0.1
        lim_max = max(np.nanmax(DAP_width_lower_per_cell), np.nanmax(DAP_width_greater_per_cell)) + 0.1
        pl.plot(np.arange(lim_min - 0.1, lim_max + 0.1, 0.1), np.arange(lim_min - 0.1, lim_max + 0.1, 0.1),
                color='0.5', linestyle='--')
        pl.plot(DAP_width_lower_per_cell, DAP_width_greater_per_cell, 'ok')
        for cell_idx in range(len(cell_ids)):
            pl.annotate(cell_ids[cell_idx], xy=(DAP_width_lower_per_cell[cell_idx] + 0.05,
                                                DAP_width_greater_per_cell[cell_idx] + 0.05))
        pl.title('DAP width')
        pl.xlabel(labels[1])
        pl.ylabel(labels[0])
        pl.tight_layout()
        if use_mean_or_std == 'mean':
            pl.savefig(os.path.join(save_dir_img, 'DAP_width_conditioned_on_mean_v.png'))
        elif use_mean_or_std == 'std':
            pl.savefig(os.path.join(save_dir_img, 'DAP_width_conditioned_on_std_v.png'))
        pl.show()