from __future__ import division
import numpy as np
import os
from load import load_full_runs
from grid_cell_stimuli.remove_APs import remove_APs
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_spectrum, plot_v_ramp_theta
from grid_cell_stimuli.downsample import antialias_and_downsample
from scipy.signal import spectrogram
import matplotlib.pyplot as pl
from spatial_firing_rate import get_spatial_firing_rate, identify_firing_fields, get_start_end_idxs_in_out_field_in_time


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/theta_and_ramp'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]

    # parameters
    cutoff_ramp = 3  # Hz
    cutoff_theta_low = 5  # Hz
    cutoff_theta_high = 11  # Hz
    transition_width = 1  # Hz
    ripple_attenuation = 60.0  # db
    params = {'cutoff_ramp': cutoff_ramp, 'cutoff_theta_low': cutoff_theta_low,
              'cut_off_theta_high': cutoff_theta_high, 'transition_width': transition_width,
              'ripple_attenuation': ripple_attenuation}

    max_ramp = []
    min_ramp = []
    max_theta = []
    min_theta = []

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5

        # remove APs
        v_APs_removed = remove_APs(v, t, AP_threshold, t_before=3, t_after=6)

        # downsample
        cutoff_freq = 2000  # Hz
        v_downsampled, t_downsampled, filter = antialias_and_downsample(v_APs_removed, dt, ripple_attenuation=60.0,
                                                                        transition_width=5.0,
                                                                        cutoff_freq=cutoff_freq,
                                                                        dt_new_max=1. / cutoff_freq * 1000)
        dt_downsampled = t_downsampled[1] - t_downsampled[0]

        # get ramp and theta
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v_downsampled, dt_downsampled,
                                                                                  ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high,
                                                                                  pad_if_to_short=True)

        plot_v_ramp_theta(v_downsampled, t_downsampled, ramp, theta, t_ramp_theta, None)
        #plot_spectrum(v_downsampled, ramp, theta, dt, None)

        cutoff_freq = 200  # Hz
        v_downsampled, t_downsampled, filter = antialias_and_downsample(v_APs_removed, dt, ripple_attenuation=60.0,
                                                                        transition_width=5.0,
                                                                        cutoff_freq=cutoff_freq,
                                                                        dt_new_max=1. / cutoff_freq * 1000)
        dt_downsampled = t_downsampled[1] - t_downsampled[0]
        frequencies, t_spec, spectogram = spectrogram(v_downsampled, fs=1 / dt_downsampled * 1000)
        t_spec *= 1000
        theta_range_bool = np.logical_and(cutoff_theta_low <= frequencies, frequencies <= cutoff_theta_high)

        pl.figure()
        pl.pcolormesh(t_spec, frequencies[theta_range_bool], spectogram[theta_range_bool, :])
        pl.ylabel('Frequency (Hz)')
        pl.xlabel('Time (ms)')
        #pl.ylim(0, 40)
        pl.colorbar()
        pl.tight_layout()
        #pl.show()
        
        # plot power spectrum for in-field vs out-field
        spatial_firing_rate, positions, loc_spikes = get_spatial_firing_rate(v, t, y_pos, pos_t, h=3,
                                                                             AP_threshold=AP_threshold, bin_size=0.5,
                                                                             track_len=np.max(y_pos))
        in_field_idxs_per_field, out_field_idxs_per_field = identify_firing_fields(spatial_firing_rate,
                                                                                   fraction_from_peak_rate=0.10)
        start_end_idx_in_field, start_end_idx_out_field = get_start_end_idxs_in_out_field_in_time(t_spec,
                                                                                                  positions, y_pos,
                                                                                                  pos_t,
                                                                                                  in_field_idxs_per_field,
                                                                                                  out_field_idxs_per_field)

        spectograms_in_field = []
        for start_idx, end_idx in start_end_idx_in_field:
            spectograms_in_field.append(spectogram[:, start_idx:end_idx])
        spectograms_in_field = np.hstack(spectograms_in_field)
        spectogram_in_field_avg = np.mean(spectograms_in_field, axis=1)
        
        spectograms_out_field = []
        for start_idx, end_idx in start_end_idx_out_field:
            spectograms_out_field.append(spectogram[:, start_idx:end_idx])
        spectograms_out_field = np.hstack(spectograms_out_field)
        spectogram_out_field_avg = np.mean(spectograms_out_field, axis=1)

        pl.figure()
        pl.plot(frequencies, spectogram_in_field_avg, 'orange', label='in-field')
        pl.plot(frequencies, spectogram_out_field_avg, 'b', label='out-field')
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('Average Power')
        pl.legend()
        pl.tight_layout()
        pl.show()