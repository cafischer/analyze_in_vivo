from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from load import load_full_runs
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from spatial_firing_rate import get_spatial_firing_rate, identify_firing_fields, get_start_end_idxs_in_out_field_in_time
from scipy.interpolate import interp1d
from grid_cell_stimuli.remove_APs import remove_APs


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape/traces'
    data_dir = '../data/'
    #cell_ids = ["20120213_12213"]
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    t_before = 3
    t_after = 6

    for cell_id in cell_ids:
        print cell_id
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5  # not capture all APs but seems
                                                                               # to be a good estimate

        # # detrend
        # cutoff_freq = 2000  # Hz
        # dt_new_max = 1. / cutoff_freq * 1000  # ms
        # transition_width = 5.0  # Hz
        # ripple_attenuation = 60.0  # db
        # v_downsampled, t_downsampled, filter = antialias_and_downsample(remove_APs(v, t, AP_threshold, t_before, t_after),
        #                                                                 dt, ripple_attenuation, transition_width,
        #                                                                 cutoff_freq, dt_new_max)
        # cutoff_ramp = 3  # Hz
        # cutoff_theta_low = 5  # Hz
        # cutoff_theta_high = 11  # Hz
        # transition_width = 1  # Hz
        # ripple_attenuation = 60.0  # db
        # dt = t_downsampled[1] - t_downsampled[0]
        # ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v_downsampled, dt, ripple_attenuation,
        #                                                                           transition_width, cutoff_ramp,
        #                                                                           cutoff_theta_low,
        #                                                                           cutoff_theta_high,
        #                                                                           pad_if_to_short=True)
        # ramp -= np.mean(v)
        # v_detrend = v - interp1d(t_downsampled, ramp)(t)
        # AP_threshold = np.min(v_detrend) + np.abs(np.max(v_detrend) - np.min(v_detrend)) * (1./2)
        # print AP_threshold

        # only use v out of field
        spatial_firing_rate, positions, loc_spikes = get_spatial_firing_rate(v, t, y_pos, pos_t, h=3,
                                                                             AP_threshold=AP_threshold, bin_size=0.5,
                                                                             track_len=np.max(y_pos))

        onsets = get_AP_onset_idxs(v, AP_threshold)
        fig, ax = pl.subplots(3, 1, sharex=True)
        ax[0].plot(t, v, 'k')
        ax[0].plot(t, np.ones(len(t))*AP_threshold, 'r')
        ax[0].plot(t[onsets], v[onsets], 'or')
        ax[1].plot(pos_t, y_pos)
        ax[2].plot(pos_t, x_pos)
        ax[0].set_ylabel('Membrane \nPotential (mV)', fontsize=14)
        ax[2].set_xlabel('Time (ms)', fontsize=14)

        # identify in and out field regions
        in_field_idxs_per_field, out_field_idxs_per_field = identify_firing_fields(spatial_firing_rate,
                                                                                   fraction_from_peak_rate=0.10)

        fig, ax = pl.subplots(2, 1, sharex=True)
        ax[0].plot(interp1d(pos_t, y_pos)(t[:np.where(t >= pos_t[-1])[0][0]]), v[:np.where(t >= pos_t[-1])[0][0]], 'k')
        ax[0].plot(loc_spikes, np.ones(len(loc_spikes))*np.max(v), 'or')
        ax[1].plot(positions, spatial_firing_rate, 'k')
        for field_idxs in in_field_idxs_per_field:
            ax[1].plot(positions[field_idxs], np.zeros(len(field_idxs)), 'orange')
        for field_idxs in out_field_idxs_per_field:
            ax[1].plot(positions[field_idxs], np.zeros(len(field_idxs)), 'b')
        ax[0].set_ylabel('Membrane \nPotential (mV)', fontsize=14)
        ax[1].set_xlabel('Position (cm)', fontsize=14)
        pl.xlim(0, np.max(y_pos))
        if y_pos[-1] < y_pos[0]:
            pl.gca().invert_xaxis()
        #pl.show()

        # get membrane potential for out fields
        start_end_idx_in_field, start_end_idx_out_field = get_start_end_idxs_in_out_field_in_time(t, positions, y_pos,
                                                                                                  pos_t,
                                                                                                  in_field_idxs_per_field,
                                                                                                  out_field_idxs_per_field)
        pl.figure()
        for start_idx, end_idx in start_end_idx_in_field:
            pl.plot(t[start_idx:end_idx], v[start_idx:end_idx], 'orange')
        for start_idx, end_idx in start_end_idx_out_field:
            pl.plot(t[start_idx:end_idx], v[start_idx:end_idx], 'b')
        #pl.show()

        # remove APs
        v_APs_removed = []
        for start_idx, end_idx in start_end_idx_out_field:
            v_APs_removed.append(remove_APs(v[start_idx:end_idx], t[start_idx:end_idx], AP_threshold, t_before, t_after))

        # histogram of v out field
        v_concatenated = np.concatenate(v_APs_removed)

        print 'Mean: %.2f' % np.mean(v_concatenated)
        print 'Std: %.2f' % np.std(v_concatenated)

        pl.figure()
        pl.hist(v_concatenated, bins=100)
        pl.xlabel('Membrane Potential (mV)', fontsize=16)
        pl.ylabel('Count', fontsize=16)
        pl.show()


# TODO:
# 12213 has one firing field less because one is discarded as it has only 14 (not 16 bins)
# 11513 2nd field missing because one is discarded as it has only 5 (not 16 bins)
# 11910b seems to have been cut in the middle beforehand or not