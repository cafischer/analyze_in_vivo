from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_sta_median, plot_APs
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/DAP_deflection'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'stellate_layer2'  # 'stellate_layer2'  #pyramidal_layer2
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
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

        # get DAP deflections
        DAP_deflections = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
            spike_characteristics_dict['AP_threshold'] = AP_thresholds[cell_id]
            spike_characteristics_dict['order_fAHP_min'] = 0.2
            spike_characteristics_dict['fAHP_interval'] = 3.0
            spike_characteristics_dict['DAP_interval'] = 4.0
            DAP_deflections[i] = get_spike_characteristics(v_AP, t_AP, ['DAP_deflection'], v_AP[0],
                                                           std_idx_times=(0, 1), check=False,
                                                           **spike_characteristics_dict)[0]
        DAP_deflections[np.isnan(DAP_deflections)] = 0
        DAP_deflections[DAP_deflections < 0] = 0
        half = np.percentile(DAP_deflections, 50)
        v_APs_greater = v_APs[DAP_deflections > half]
        v_APs_lower = v_APs[DAP_deflections <= half]

        # STA
        sta_mean_greater, sta_std_greater = get_sta(v_APs_greater)
        sta_mean_lower, sta_std_lower = get_sta(v_APs_lower)

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        pl.figure()
        pl.plot(t_AP, sta_mean_greater, 'r', label='High DAP deflection')
        pl.fill_between(t_AP, sta_mean_greater + sta_std_greater, sta_mean_greater - sta_std_greater,
                        facecolor='r', alpha=0.5)
        pl.plot(t_AP, sta_mean_lower, 'b', label='Low DAP deflection')
        pl.fill_between(t_AP, sta_mean_lower + sta_std_lower, sta_mean_lower - sta_std_lower,
                        facecolor='b', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'sta.png'))

        v_APs_plots_greater = v_APs_greater[np.random.randint(0, len(v_APs_greater), 20)]  # reduce to lower number
        v_APs_plots_lower = v_APs_lower[np.random.randint(0, len(v_APs_lower), 20)]  # reduce to lower number

        pl.figure()
        for i, v_AP in enumerate(v_APs_plots_lower):
            pl.plot(t_AP, v_AP, 'b', alpha=0.5, label='Low DAP deflection' if i == 0 else '')
        for i, v_AP in enumerate(v_APs_plots_greater):
            pl.plot(t_AP, v_AP, 'r', alpha=0.5, label='High DAP deflection' if i == 0 else '')
        pl.ylabel('Membrane potential (mV)')
        pl.xlabel('Time (ms)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'v_APs.png'))
        #pl.show()