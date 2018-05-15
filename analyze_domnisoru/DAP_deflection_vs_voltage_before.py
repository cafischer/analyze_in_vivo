from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from cell_characteristics import to_idx
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from sklearn import linear_model
from scipy.stats import pearsonr
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/DAP_deflection_vs_v_before'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'pyramidal_layer2'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}
    param_list = ['Vm_ljpc']

    # parameters
    in_field = False
    out_field = False
    after_AP = 20
    offset = 2
    dur = 10
    before_AP = dur+offset+offset
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, folder_field[(in_field, out_field)],
                                cell_type)

    #
    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx_sta = to_idx(before_AP, dt)
        after_AP_idx_sta = to_idx(after_AP, dt)

        # get APs
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
            DAP_deflections[i] = get_spike_characteristics(v_AP[to_idx(offset+dur, dt):],
                                                           t_AP[to_idx(offset+dur, dt):],
                                                           ['DAP_deflection'], v_AP[0],
                                                           std_idx_times=(0, 1), check=False,
                                                           **spike_characteristics_dict)[0]
        DAP_deflections[np.isnan(DAP_deflections)] = 0
        DAP_deflections[DAP_deflections < 0] = 0

        v_before = np.mean(v_APs[:, to_idx(offset, dt):to_idx(dur+offset, dt)], 1)

        # pl.figure()
        # for i in range(len(v_APs)):
        #     pl.plot(t_AP[to_idx(offset, dt):to_idx(dur+offset, dt)],
        #             v_APs[i, to_idx(offset, dt):to_idx(dur+offset, dt)])
        # pl.show()

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        regr = linear_model.LinearRegression()
        regr.fit(np.array([DAP_deflections]).T, np.array([v_before]).T)
        a = regr.coef_[0]
        b = regr.intercept_[0]

        pl.figure()
        pl.title('corr: %.2f \n p-value: %.2f' % (pearsonr(DAP_deflections, v_before)))
        pl.plot(DAP_deflections, v_before, 'ko')
        pl.plot(DAP_deflections, a * DAP_deflections + b, 'r',
                label='slope: %.2f' % a)
        pl.xlabel('DAP deflection (mV)')
        pl.ylabel('Mean voltage '+str(dur)+' ms \nbefore AP max (mV)')
        pl.tight_layout()
        pl.legend()
        pl.savefig(os.path.join(save_dir_cell, 'DAP_deflection_vs_voltage_before_'+str(dur)+'ms.png'))
        #pl.show()