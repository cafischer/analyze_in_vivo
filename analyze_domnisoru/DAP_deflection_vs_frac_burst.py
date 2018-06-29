from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, get_sta_median
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/DAP_deflection_vs_AP_amp'
    save_dir_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_spat_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    burst_ISI = 8  # ms
    use_AP_max_idxs_domnisoru = True
    before_AP_sta = 25
    after_AP_sta = 25
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fraction_burst = np.load(os.path.join(save_dir_burst, cell_type, 'fraction_burst.npy'))
    spatial_info = np.load(os.path.join(save_dir_spat_info, cell_type, 'spatial_info.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, cell_type, 'peak_ISI_hist.npy'))
    peak_auto_corr = np.load(os.path.join(save_dir_auto_corr, cell_type, 'peak_auto_corr_50.npy'))

    #
    DAP_deflection_per_cell = np.zeros(len(cell_ids))
    DAP_width_per_cell = np.zeros(len(cell_ids))
    DAP_time_per_cell = np.zeros(len(cell_ids))
    AP_width_per_cell = np.zeros(len(cell_ids))
    frac_ISI_per_cell = np.zeros(len(cell_ids))

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

        v_APs = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        if v_APs is None:
            continue

        # DAP_deflection and AP max from STA
        sta_mean, sta_std = get_sta(v_APs)
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx_sta
        spike_characteristics_dict['AP_onset'] = before_AP_idx_sta - to_idx(1, dt)
        AP_width_per_cell[i],  DAP_deflection_per_cell[i], \
        DAP_width_per_cell[i], DAP_time_per_cell[i] = get_spike_characteristics(sta_mean, t_AP,
                                                                        ['AP_width', 'DAP_deflection',
                                                                         'DAP_width', 'DAP_time'],
                                                                        sta_mean[0],
                                                                        check=False, **spike_characteristics_dict)

    good_cell_indicator = AP_width_per_cell <= 0.75
    DAP_deflection_per_cell = DAP_deflection_per_cell[good_cell_indicator]
    DAP_time_per_cell = DAP_time_per_cell[good_cell_indicator]
    fraction_burst = fraction_burst[good_cell_indicator]
    spatial_info = spatial_info[good_cell_indicator]
    peak_ISI_hist = peak_ISI_hist[good_cell_indicator]
    peak_auto_corr = peak_auto_corr[good_cell_indicator]
    cell_ids = np.array(cell_ids)[good_cell_indicator]

    pl.figure()
    pl.plot(np.zeros(len(cell_ids))[DAP_deflection_per_cell == 0],
            fraction_burst[DAP_deflection_per_cell == 0], 'o', color='0.5')
    pl.plot(np.ones(len(cell_ids))[DAP_deflection_per_cell > 0],
            fraction_burst[DAP_deflection_per_cell > 0], 'o', color='0.5')
    for cell_idx in range(len(cell_ids)):
        pl.annotate(cell_ids[cell_idx], xy=((DAP_deflection_per_cell[cell_idx] > 0).astype(int), fraction_burst[cell_idx]))
    pl.plot(0, np.mean(fraction_burst[DAP_deflection_per_cell == 0]), 'ok')
    pl.plot(1, np.mean(fraction_burst[DAP_deflection_per_cell > 0]), 'ok')
    pl.xlim(-1, 2)
    pl.xticks([0, 1])
    pl.ylabel('Fraction ISI < 8 ms')
    pl.xlabel('DAP deflection (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_frac_burst.png'))

    pl.figure()
    pl.plot(np.zeros(len(cell_ids))[DAP_deflection_per_cell == 0],
            spatial_info[DAP_deflection_per_cell == 0], 'o', color='0.5')
    pl.plot(np.ones(len(cell_ids))[DAP_deflection_per_cell > 0],
            spatial_info[DAP_deflection_per_cell > 0], 'o', color='0.5')
    for cell_idx in range(len(cell_ids)):
        pl.annotate(cell_ids[cell_idx], xy=((DAP_deflection_per_cell[cell_idx] > 0).astype(int), spatial_info[cell_idx]))
    pl.plot(0, np.mean(spatial_info[DAP_deflection_per_cell == 0]), 'ok')
    pl.plot(1, np.mean(spatial_info[DAP_deflection_per_cell > 0]), 'ok')
    pl.ylabel('Spatial information')
    pl.xlabel('DAP deflection (mV)')
    pl.xlim(-1, 2)
    pl.xticks([0, 1])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_spatial_info.png'))

    peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak
    pl.figure()
    pl.plot(DAP_time_per_cell, peak_ISI_hist, 'o', color='0.5')
    pl.plot(DAP_time_per_cell, peak_auto_corr, 'o', color='r')
    for cell_idx in range(len(cell_ids)):
        pl.annotate(cell_ids[cell_idx], xy=(DAP_time_per_cell[cell_idx], peak_ISI_hist[cell_idx]))
    pl.xlim(0, 10)
    pl.ylim(0, 10)
    pl.ylabel('Peak of ISI hist. (ms)')
    pl.xlabel('DAP time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_time_vs_ISI_peak.png'))
    pl.show()