from __future__ import division
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.spike_phase import get_spike_phases, plot_phase_hist, plot_phase_vs_position_per_run


if __name__ == '__main__':
    folder = 'schmidthieber'
    save_dir = './results/' + folder + '/spike_phase'
    save_dir_data = './results/' + folder + '/downsampled'
    save_dir_pos = './results/' + folder + '/data'
    save_dir_theta = './results/' + folder + '/ramp_and_theta'

    # over all field crossings
    for i, file_name in enumerate(os.listdir(save_dir_data)):

        # load
        v = np.load(os.path.join(save_dir_data, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        dt = t[1] - t[0]
        position = np.load(os.path.join(save_dir_pos, file_name, 'position.npy'))
        pos_t = np.load(os.path.join(save_dir_pos, file_name, 'pos_t.npy'))
        position = np.interp(t, pos_t, position)
        track_len = np.max(position)
        theta = np.load(os.path.join(save_dir_theta, file_name, 'theta.npy'))

        # parameter
        order = int(round(20 / dt))
        dist_to_AP = int(round(200 / dt))

        # extract phase
        AP_threshold = np.max(v) - np.abs((np.min(v) - np.max(v)) / 3)
        AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
        phases_pos = position[AP_onsets]
        phases = get_spike_phases(AP_onsets, t, theta, order, dist_to_AP)

        not_nan = np.logical_not(np.isnan(phases))
        phases_not_nan = phases[not_nan]
        phases_pos_not_nan = phases_pos[not_nan]

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        np.save(os.path.join(save_dir_cell_field_crossing, 'phases.npy'), phases)
        np.save(os.path.join(save_dir_cell_field_crossing, 'phases_pos.npy'), phases_pos)

        plot_phase_hist(phases_not_nan, save_dir_cell_field_crossing)
        run_start_idx = [0, len(position)]
        plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, position, track_len, run_start_idx,
                                       save_dir_cell_field_crossing)