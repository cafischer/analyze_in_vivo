import numpy as np
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta
from grid_cell_stimuli.downsample import antialias_and_downsample
from scipy.interpolate import interp1d


def detrend(v, t, cutoff_freq=5):
    cutoff_freq_downsampling = 2000  # Hz
    dt_new_max = 1. / cutoff_freq_downsampling * 1000  # ms
    transition_width = 5.0  # Hz
    ripple_attenuation = 60.0  # db
    dt = t[1] - t[0]
    v_downsampled, t_downsampled, filter = antialias_and_downsample(v, dt, ripple_attenuation, transition_width,
                                                                    cutoff_freq_downsampling, dt_new_max)

    transition_width = 1  # Hz
    ripple_attenuation = 60.0  # db
    dt_downsampled = t_downsampled[1] - t_downsampled[0]
    ramp, _, t_ramp_theta, _, _ = get_ramp_and_theta(v_downsampled, dt_downsampled,
                                                     ripple_attenuation,
                                                     transition_width, cutoff_freq,
                                                     cutoff_theta_low=1,
                                                     cutoff_theta_high=2,
                                                     pad_if_to_short=True)
    ramp -= np.mean(v)
    v_detrend = v - interp1d(t_downsampled, ramp)(t)
    return v_detrend