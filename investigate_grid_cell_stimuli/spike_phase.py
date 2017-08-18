from __future__ import division
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.spike_phase import get_spike_phases, plot_phase_hist, plot_phase_vs_position_per_run, \
    compute_phase_precession, plot_phase_precession


if __name__ == '__main__':
    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/spike_phase'
    save_dir_data = '../results/' + folder + '/data'
    save_dir_downsample = '../results/' + folder + '/downsampled'
    save_dir_theta = '../results/' + folder + '/ramp_and_theta'

    file_names = os.listdir(save_dir_data)
    discard_files = ['20110910_11910b_0_0', '20120213_12213_1_0', '20111207_11d07c_0_0']
    reverse_files = ['20111213_11d13b_0_0', '20110910_11910b_1_0', '20101031_10o31c_0_0', '20101031_10o31c_1_0']

    # over all field crossings
    phases_all = []
    slope_all = []
    for i, file_name in enumerate(file_names):
        if file_name in discard_files:
            print 'discard', file_name
            continue

        # load
        v = np.load(os.path.join(save_dir_downsample, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_downsample, file_name, 't.npy'))
        dt = t[1] - t[0]
        position = np.load(os.path.join(save_dir_data, file_name, 'position.npy'))
        pos_t = np.load(os.path.join(save_dir_data, file_name, 'pos_t.npy'))
        position = np.interp(t, pos_t, position)
        track_len = np.max(position)
        theta = np.load(os.path.join(save_dir_theta, file_name, 'theta.npy'))
        assert len(v) == len(theta)

        if file_name in reverse_files:
            print 'reverse', file_name
            position_reversed = np.max(position) - position  # do as if it started from 0 and went to end of track
            # pl.figure()
            # pl.plot(t, position, 'k')
            # pl.plot(t, position_new, 'k')
            # pl.ylabel('Position (cm)', fontsize=16)
            # pl.xlabel('Time (ms)', fontsize=16)
            # pl.show()
            position = position_reversed

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
        phases_all.extend(phases_not_nan)

        # phase precession
        slope, intercept, best_shift = compute_phase_precession(phases_not_nan, phases_pos_not_nan)
        slope_all.append(slope)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        np.save(os.path.join(save_dir_cell_field_crossing, 'phases.npy'), phases)
        np.save(os.path.join(save_dir_cell_field_crossing, 'phases_pos.npy'), phases_pos)

        plot_phase_hist(phases_not_nan, save_dir_cell_field_crossing)
        run_start_idx = [0, len(position)]
        plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx,
                                       save_dir_cell_field_crossing)
        plot_phase_precession(phases_not_nan, phases_pos_not_nan, slope, intercept, best_shift,
                              save_dir_cell_field_crossing)
    plot_phase_hist(phases_all, save_dir)
    print 'Mean absolute slope: ', np.mean(np.abs(slope))