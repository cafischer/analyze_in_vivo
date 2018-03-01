from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load import load_full_runs
from analyze_in_vivo.investigate_grid_cell_stimuli import detrend
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
import matplotlib.pyplot as pl
from itertools import combinations

pl.style.use('paper')

if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/plot_APs'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    cell_ids = ["20110910_11910b"]

    # parameters
    do_detrend = True
    before_AP = 0
    after_AP = 25
    cut_before_AP = before_AP
    cut_after_AP = after_AP - 25

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        cut_before_AP_idx = to_idx(cut_before_AP, dt)
        cut_after_AP_idx = to_idx(cut_after_AP, dt)

        # detrend
        if do_detrend:
            v_detrend = detrend(v, t, cutoff_freq=5)
            AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5

        # find all spikes
        v_APs = []
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)

        if len(onset_idxs) > 0:
            onset_idxs = np.insert(onset_idxs, len(onset_idxs), len(v))
            AP_max_idxs = [get_AP_max_idx(v, onset_idx, onset_next_idx) for (onset_idx, onset_next_idx)
                           in zip(onset_idxs[:-1], onset_idxs[1:])]
            # pl.figure()
            # pl.plot(t, v)
            # pl.plot(t[AP_max_idxs], v[AP_max_idxs], 'or')
            # pl.show()

            # take window around each spike
            for AP_max_idx in AP_max_idxs:
                if (AP_max_idx is not None
                        and AP_max_idx - before_AP_idx >= 0
                        and AP_max_idx + after_AP_idx + 1 <= len(v)):  # able to draw window
                    v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]
                    if len(get_AP_onset_idxs(v_AP,
                                             AP_threshold)) == 0:  # no bursts (1st AP should not be detected as it starts from the max)
                        if do_detrend:
                            if cell_id == '20120213_12213':
                                if not np.any(
                                        np.diff(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                                        > 5):  # check for measurement errors
                                    v_APs.append(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                            else:
                                v_APs.append(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                        else:
                            if cell_id == '20120213_12213':
                                if not np.any(np.diff(v_AP) > 5):  # check for measurement errors
                                    v_APs.append(v_AP)
                            else:
                                v_APs.append(v_AP)
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # plot
        if do_detrend:
            save_dir_img = os.path.join(save_dir, 'detrended', cell_id)
        else:
            save_dir_img = os.path.join(save_dir, 'not_detrended', cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        fig, axes = pl.subplots(17, 10,    # int(np.floor(np.sqrt(len(v_APs)))), int(np.ceil(np.sqrt(len(v_APs)))),
                                sharex='all', sharey='all', figsize=(21, 29.7))
        for i, ax in enumerate(axes.flatten()):
            if i < len(v_APs):
                ax.plot(t_AP, v_APs[i, :])
        #pl.ylabel('Membrane potential (mV)', fontsize=16)
        #pl.xlabel('Time (ms)', fontsize=16)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_APs.pdf'))
        pl.show()