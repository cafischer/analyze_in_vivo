from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load import load_field_crossings, get_stellate_info
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.spike_phase import get_spike_phases
from cell_characteristics import to_idx
from scipy.stats import circmean


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/ISI_phases'
    save_dir_data = '../results/' + folder + '/data'
    save_dir_theta = '../results/' + folder + '/ramp_and_theta'
    save_dir_downsample = '../results/' + folder + '/downsampled'

    # parameter
    AP_thresholds = [-40, -40, -50, -30, -40, -50]
    bins = np.arange(0, 200, 2)
    short_ISI_threshold = 15  # ms

    # over cells
    file_names = os.listdir(save_dir_data)

    # over all field crossings
    phases_short_all = []
    phases_single_all = []
    for i, file_name in enumerate(file_names):

        # load
        v = np.load(os.path.join(save_dir_downsample, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_downsample, file_name, 't.npy'))

        AP_threshold = np.max(v) - np.abs((np.min(v) - np.max(v)) / 3)

        # ISI
        AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
        ISIs = np.diff(t[AP_onsets])
        ISIs_short = np.array(ISIs <= short_ISI_threshold, dtype=int)
        ISIs_short_first_in_burst = np.where(np.diff(ISIs_short) == 1)[0] + 1
        if ISIs_short[0] == 1:
            ISIs_short_first_in_burst = np.concatenate((np.array([0]), ISIs_short_first_in_burst))

        ISIs_short = np.concatenate((ISIs_short, np.array([0])))
        ISIs_short[np.where(ISIs_short)[0] + 1] = 1
        ISIs_single = np.where(ISIs_short == 0)

        # phase of ISI
        dt = t[1] - t[0]
        theta = np.load(os.path.join(save_dir_theta, file_name, 'theta.npy'))
        order = to_idx(20, dt)
        dist_to_AP = to_idx(200, dt)
        phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)
        phases_short_ISI = phases[ISIs_short_first_in_burst]
        phases_short_ISI = phases_short_ISI[~np.isnan(phases_short_ISI)]
        phases_single = phases[ISIs_single]
        phases_single = phases_single[~np.isnan(phases_single)]
        phases_short_all.extend(phases_short_ISI)
        phases_single_all.extend(phases_single)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        pl.figure()
        pl.plot(t, v, 'k')
        pl.plot(t[AP_onsets[ISIs_short_first_in_burst]], v[AP_onsets[ISIs_short_first_in_burst]], 'or')
        pl.plot(t[AP_onsets[ISIs_single]], v[AP_onsets[ISIs_single]], 'ob')
        pl.savefig(os.path.join(save_dir_cell_field_crossing, 'v_with_ISI_marked.svg'))
        # pl.show()

        pl.figure()
        pl.hist(phases_short_ISI, bins=np.arange(0, 360 + 10, 10), color='r', alpha=0.5, label='First AP where ISI <= '+str(short_ISI_threshold))
        pl.hist(phases_single, bins=np.arange(0, 360 + 10, 10), color='b', alpha=0.5, label='Single AP')
        pl.axvline(circmean(phases_short_ISI, 360, 0), color='r', linewidth=2)
        pl.axvline(circmean(phases_single, 360, 0), color='b', linewidth=2)
        pl.xlabel('Phase ($^{\circ}$)')
        pl.ylabel('Count')
        pl.legend()
        pl.xlim(0, 360)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell_field_crossing, 'ISI_phase_hist.svg'))
        # pl.show()

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.hist(phases_short_all, bins=np.arange(0, 360 + 10, 10), color='r', alpha=0.5, label='First AP where ISI <= '+str(short_ISI_threshold))
    pl.hist(phases_single_all, bins=np.arange(0, 360 + 10, 10), color='b', alpha=0.5, label='Single AP')
    pl.axvline(circmean(phases_short_all, 360, 0), color='r', linewidth=2)
    pl.axvline(circmean(phases_single_all, 360, 0), color='b', linewidth=2)
    pl.xlabel('Phase ($^{\circ}$)')
    pl.ylabel('Count')
    pl.xlim(0, 360)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'ISI_phase_hist.svg'))
    pl.show()