from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from load import load_full_runs
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from grid_cell_stimuli.remove_APs import remove_APs
from spatial_firing_rate import get_spatial_firing_rate
from scipy.interpolate import interp1d
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta
from grid_cell_stimuli.downsample import antialias_and_downsample

if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape/traces'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    t_before = 3
    t_after = 6

    for cell_id in cell_ids:
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + np.abs(np.max(v) - np.min(v)) * (1./2)
        print AP_threshold

        # detrend
        cutoff_freq = 2000  # Hz
        dt_new_max = 1. / cutoff_freq * 1000  # ms
        transition_width = 5.0  # Hz
        ripple_attenuation = 60.0  # db
        v_downsampled, t_downsampled, filter = antialias_and_downsample(remove_APs(v, t, AP_threshold, t_before, t_after),
                                                                        dt, ripple_attenuation, transition_width,
                                                                        cutoff_freq, dt_new_max)
        cutoff_ramp = 3  # Hz
        cutoff_theta_low = 5  # Hz
        cutoff_theta_high = 11  # Hz
        transition_width = 1  # Hz
        ripple_attenuation = 60.0  # db
        dt = t_downsampled[1] - t_downsampled[0]
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v_downsampled, dt, ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high,
                                                                                  pad_if_to_short=True)
        ramp -= np.mean(v)
        v_detrend = v - interp1d(t_downsampled, ramp)(t)
        AP_threshold = np.min(v_detrend) + np.abs(np.max(v_detrend) - np.min(v_detrend)) * (1./2)
        print AP_threshold

        # only use v out of field
        spatial_firing_rate, positions, loc_spikes = get_spatial_firing_rate(v_detrend, t, y_pos, pos_t, h=3,
                                                                             AP_threshold=AP_threshold, bin_size=0.5,
                                                                             track_len=np.max(y_pos))

        onsets = get_AP_onset_idxs(v_detrend, AP_threshold)
        fig, ax = pl.subplots(3, 1, sharex=True)
        #ax[0].plot(t, v, 'k')
        ax[0].plot(t, v_detrend, 'orange')
        ax[0].plot(t, interp1d(t_downsampled, ramp)(t) + np.mean(v), 'g')
        ax[0].plot(t, np.ones(len(t))*AP_threshold, 'r')
        ax[0].plot(t[onsets], v[onsets], 'or')
        ax[1].plot(pos_t, y_pos)
        ax[2].plot(pos_t, x_pos)
        ax[0].set_ylabel('Membrane \nPotential (mV)', fontsize=14)
        ax[2].set_xlabel('Time (ms)', fontsize=14)

        fig, ax = pl.subplots(2, 1, sharex=True)
        ax[0].plot(interp1d(pos_t, y_pos)(t[:np.where(t >= pos_t[-1])[0][0]]), v[:np.where(t >= pos_t[-1])[0][0]], 'k')
        ax[0].plot(loc_spikes, np.ones(len(loc_spikes))*np.max(v), 'or')
        ax[1].plot(positions, spatial_firing_rate)
        ax[0].set_ylabel('Membrane \nPotential (mV)', fontsize=14)
        ax[1].set_xlabel('Position (cm)', fontsize=14)
        pl.xlim(0, np.max(y_pos))
        if y_pos[-1] < y_pos[0]:
            pl.gca().invert_xaxis()
        pl.show()

        # pl.figure()
        # pl.plot(positions, spatial_firing_rate)
        # pl.xlabel('Position (cm)')
        # pl.ylabel('Firing rate')
        # pl.tight_layout()
        # pl.show()

        # remove APs
        #TODO: v_APs_removed = remove_APs(v, t, AP_threshold, t_before, t_after)
        #v = v_APs_removed

        print 'Mean: %.2f' % np.mean(v)
        print 'Std: %.2f' % np.std(v)

        pl.figure()
        pl.plot(t, v, 'k')
        pl.ylabel('Membrane Potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)

        pl.figure()
        pl.hist(v, bins=100)
        pl.xlabel('Membrane Potential (mV)', fontsize=16)
        pl.ylabel('Count', fontsize=16)
        pl.show()

        # TODO: AP threshold adaptive?