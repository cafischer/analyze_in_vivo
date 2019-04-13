from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from cell_characteristics import to_idx


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/prob_2nd_spike'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir_theta_ramp = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/theta_ramp'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'pyramidal_layer3'
    save_dir_sta = os.path.join('/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA',
                                'not_detrended', 'all', cell_type)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    velocity_threshold = 1  # cm/sec
    max_ISI = 25
    use_AP_max_idxs_domnisoru = True

    for cell_id in cell_ids:
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        sta_mean = np.load(os.path.join(save_dir_sta, cell_id, 'sta_mean.npy'))
        before_AP_sta = 25
        after_AP_sta = 25
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)
        sta_mean = sta_mean[before_AP_idx_sta:]

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        starts_burst, ends_burst = get_starts_ends_group_of_ones(np.concatenate((ISIs <= max_ISI, np.array([False]))).astype(int))
        starts_burst = starts_burst[starts_burst < len(AP_max_idxs)-1]

        # first ISI after burst
        first_ISI_after_burst = t[AP_max_idxs[starts_burst+1]] - t[AP_max_idxs[starts_burst]]
        first_ISI_after_burst = first_ISI_after_burst[first_ISI_after_burst <= max_ISI]

        # plots
        # pl.figure()
        # pl.plot(np.arange(len(sta_mean)) * dt, sta_mean, 'k')
        # pl.hist(first_ISI_after_burst, bins=np.arange(0, max_ISI+0.25, 0.25), color='0.5')
        # pl.xlabel('Time after AP (ms)')
        # pl.ylabel('# burst APs')
        # pl.xlim(0, max_ISI)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'prob_2nd_spike.png'))

        bins = np.arange(0, max_ISI + 0.25, 0.25)
        hist, _ = np.histogram(first_ISI_after_burst, bins=bins)
        bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2.

        fig, ax = pl.subplots()
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(sta_mean)) * dt, sta_mean, 'k')
        ax2.set_ylabel('Membrane Potential (mV)')
        ax2.spines['right'].set_visible(True)
        ax.plot(bin_midpoints, hist, '-o', color='k', markersize=9 - 2.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency 2nd AP')
        ax.set_xlim(0, after_AP_sta)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'prob_2nd_spike.png'))
        pl.show()