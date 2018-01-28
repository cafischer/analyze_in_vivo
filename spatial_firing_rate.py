from __future__ import division
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl


def gauss_filter(x):
    return np.exp(-x**2)


def get_spatial_firing_rate(v, t, x_pos, pos_t, h=3, AP_threshold=0, bin_size=0.5, track_len=430):
    """

    :param v:
    :param x_pos: (cm)
    :param h: (cm)
    :param AP_threshold: (mV).
    :param bin_size: Size (cm) of spatial bins.
    :param track_len: Length (cm) of the track.
    :return: 
    """
    onsets = get_AP_onset_idxs(v, AP_threshold)
    time_spikes = t[onsets]
    bin_spikes = np.array([np.where((pos_t - t_spike) >= 0)[0][0] for t_spike in time_spikes])
    location_spikes = x_pos[bin_spikes]

    firing_rate = np.zeros(int(np.ceil(track_len / bin_size))+1)
    positions = np.arange(0, track_len+bin_size, bin_size)
    for i, pos in enumerate(positions):
        firing_rate[i] = np.sum(gauss_filter((location_spikes - pos) / h)) / np.sum(gauss_filter((x_pos - pos) / h))

    return firing_rate, positions, location_spikes

# TODO: unit firing rate?
# TODO: why x_pos not going from 0 to 430 or otherway around but stopping earlier and why once changing directions in between?

if __name__ == '__main__':
    # test spatial firing rate
    tstop = 100
    dt = 0.01
    t = np.arange(0, tstop + dt, dt)
    v = np.ones(len(t)) * -65

    spike_fun_exp = lambda s: np.exp(s / 15)
    spike_fun_lin = lambda s: s
    spike_fun_sig = lambda s: tstop * 1/(1 + np.exp(-(s-tstop/2)))

    spike_times = np.arange(10, tstop, 10)
    spike_times = spike_fun_exp(spike_times)
    for s_t in spike_times:
        v[int(round(s_t / dt)):int(round((s_t+1) / dt))] = 20
    track_len = 100
    bin_size = 0.5
    pos_fun_exp = lambda p: np.exp(p/15)
    pos_fun_log = lambda p: np.log(p) * 15
    pos_fun_lin = lambda p: p
    pos_fun_sig = lambda p: track_len * 1/(1 + np.exp(-(p-track_len/2)/10))
    x_pos = pos_fun_log(np.arange(0, track_len+bin_size, bin_size))
    pos_t = np.linspace(0, tstop, len(x_pos))

    # pl.figure()
    # pl.plot(pos_t, x_pos)
    # pl.show()

    spatial_firing_rate, positions, location_spikes = get_spatial_firing_rate(v, t, x_pos, pos_t, h=3, AP_threshold=0,
                                                                              bin_size=bin_size, track_len=track_len)

    times = positions * np.insert((np.diff(pos_t) / np.diff(x_pos)), 0, (pos_t[1] / x_pos[1]))

    pl.figure()
    pl.plot(positions, spatial_firing_rate)
    pl.plot(location_spikes, np.zeros(len(location_spikes)), 'ob')
    pl.xlabel('Position (cm)')
    pl.ylabel('Firing rate')
    pl.tight_layout()
    pl.show()