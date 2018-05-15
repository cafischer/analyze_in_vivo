from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_sta_median, plot_APs
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
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

    #
    sta_mean_per_cell = []
    sta_std_per_cell = []
    for i, cell_id in enumerate(cell_ids):
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
        if v_APs is None:
            continue

        # STA
        sta_median, sta_mad = get_sta_median(v_APs)
        sta_mean, sta_std = get_sta(v_APs)
        sta_mean_per_cell.append(sta_mean)
        sta_std_per_cell.append(sta_std)

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
        np.save(os.path.join(save_dir_cell, 'sta_mean.npy'), sta_mean)
        plot_sta(t_AP, sta_median, sta_mad, os.path.join(save_dir_cell, 'sta_median.png'))
        plot_sta(t_AP, sta_mean, sta_std, os.path.join(save_dir_cell, 'sta_mean.png'))
        plot_APs(v_APs, t_AP, os.path.join(save_dir_cell, 'v_APs.png'))
        #pl.show()

    pl.close('all')
    n_rows = 1 if len(cell_ids) <= 3 else 2
    fig_height = 4.5 if len(cell_ids) <= 3 else 9
    fig, axes = pl.subplots(n_rows, int(round(len(cell_ids)/n_rows)), sharex='all', sharey='all', figsize=(12, fig_height))
    if n_rows == 1:
        axes = np.array([axes])
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(int(round(len(cell_ids) / n_rows))):
            if cell_idx < len(cell_ids):
                axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                axes[i1, i2].plot(t_AP, sta_mean_per_cell[cell_idx], 'k')
                axes[i1, i2].fill_between(t_AP, sta_mean_per_cell[cell_idx] - sta_std_per_cell[cell_idx],
                                          sta_mean_per_cell[cell_idx] + sta_std_per_cell[cell_idx], color='k', alpha=0.5)
                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('Time (ms)')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('Membrane Potential (mV)')
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    adjust_bottom = 0.12 if len(cell_ids) <= 3 else 0.08
    pl.subplots_adjust(left=0.08, bottom=adjust_bottom)
    pl.savefig(os.path.join(save_dir_img, 'sta.png'))
    pl.show()


    #     # DAP_deflection on STA
    #     from cell_characteristics.analyze_APs import get_spike_characteristics
    #     from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
    #     import json
    #     spike_characteristics_dict = get_spike_characteristics_dict()
    #     spike_characteristics_dict['AP_threshold'] = AP_thresholds[cell_id]
    #     DAP_deflections[cell_id] = get_spike_characteristics(sta, t_AP, ['DAP_deflection'], sta[0],
    #                                                check=False, **spike_characteristics_dict)[0]
    # print DAP_deflections
    # with open(os.path.join(save_dir_img, 'not_detrended', cell_type, 'DAP_deflections.npy'), 'w') as f:
    #     json.dump(DAP_deflections, f, indent=4)