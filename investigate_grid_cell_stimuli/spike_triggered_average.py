from __future__ import division
import numpy as np
import os
from load import load_full_runs
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics import to_idx
import matplotlib.pyplot as pl
pl.style.use('paper')


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/STA'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]

    # parameters
    before_AP = 20
    after_AP = 40

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

        # find all spikes
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)

        # take window around each spike
        v_APs = []
        for onset_idx in onset_idxs:
            if onset_idx-before_AP_idx >= 0 and onset_idx+after_AP_idx+1 <= len(v):  # able to draw window
                v_AP = v[onset_idx-before_AP_idx:onset_idx+after_AP_idx+1]
                if len(get_AP_onset_idxs(v_AP, AP_threshold)) == 1:  # no bursts
                    v_APs.append(v_AP)
        v_APs = np.vstack(v_APs)

        # STA
        spike_triggered_avg = np.mean(v_APs, 0)
        spike_triggered_std = np.std(v_APs, 0)
        t_AP = np.arange(after_AP_idx+before_AP_idx+1)*dt

        print '#APs: ' + str(len(v_APs))

        # plot
        save_dir_cell = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg, 'b')
        pl.fill_between(t_AP, spike_triggered_avg+spike_triggered_std, spike_triggered_avg-spike_triggered_std,
                        facecolor='blue', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('V (mV)')
        pl.title(cell_id.split('_')[1])
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'sta.png'))
        #pl.show()

        pl.figure()
        for v_AP in v_APs:
            pl.plot(t_AP, v_AP)
        #pl.show()

        pl.figure()
        pl.plot(t, v)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'v.png'))
        pl.show()