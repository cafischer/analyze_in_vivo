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
    bin_spikes = np.digitize(time_spikes, pos_t)
    location_spikes = x_pos[bin_spikes]

    firing_rate = np.zeros(int(np.ceil(track_len / bin_size))+1)
    positions = np.arange(0, track_len+bin_size, bin_size)
    for i, pos in enumerate(positions):
        firing_rate[i] = (1000 *  # convert to sec (as integration adds time as unit)
                         np.sum(gauss_filter((location_spikes - pos) / h)) / np.sum(gauss_filter((x_pos - pos) / h)))

    return firing_rate, positions, location_spikes


def identify_firing_fields(spatial_firing_rate):
    peak_rate = np.nanmax(spatial_firing_rate)
    in_field_idxs_tmp = np.where(spatial_firing_rate > peak_rate * 0.20)[0]  # firing rate > 20% of the peak rate
    in_field_idxs_per_field_tmp = divide_idxs_into_fields(in_field_idxs_tmp)
    in_field_idxs_per_field = []
    for in_field_idxs in in_field_idxs_per_field_tmp:
        if (len(in_field_idxs) >= 16  # contiguous region of at least 16 bins
                and np.max(spatial_firing_rate[
                               in_field_idxs]) >= 1):  # peak rate >= 1Hz
            in_field_idxs_per_field.append(in_field_idxs)

    out_field_idxs = np.setdiff1d(np.arange(len(spatial_firing_rate)), np.array(flatten_list(in_field_idxs_per_field)),
                                  assume_unique=True)
    out_field_idxs_per_field = divide_idxs_into_fields(out_field_idxs)
    return in_field_idxs_per_field, out_field_idxs_per_field


def divide_idxs_into_fields(field_idxs):
    start_idxs = np.where(np.diff(field_idxs) > 1)[0] + 1
    start_idxs = np.concatenate((np.array([0]), start_idxs, np.array([len(field_idxs)])))
    fields_idxs_per_field = [field_idxs[s:e] for (s, e) in zip(start_idxs[:-1], start_idxs[1:])]
    return fields_idxs_per_field


def flatten_list(list):
    return [item for sublist in list for item in sublist]


def get_start_end_idxs_in_out_field_in_time(t, positions, y_pos, pos_t, in_field_idxs_per_field, out_field_idxs_per_field):
    y_pos_idxs = np.digitize(y_pos, positions)

    y_pos_in_field = np.array([i in flatten_list(in_field_idxs_per_field) for i in
                                y_pos_idxs])  # for each y_pos_new determine whether its out field
    idxs_per_in_field = divide_idxs_into_fields(np.arange(len(y_pos_in_field))[y_pos_in_field])

    y_pos_out_field = np.array([i in flatten_list(out_field_idxs_per_field) for i in
                                y_pos_idxs])  # for each y_pos_new determine whether its out field
    idxs_per_out_field = divide_idxs_into_fields(np.arange(len(y_pos_out_field))[y_pos_out_field])

    # y_pos_binned = positions[y_pos_idxs]  # values only as in positions
    # pl.figure()
    # pl.plot(pos_t, y_pos_binned, 'oorange')
    # pl.plot(pos_t[y_pos_out_field], y_pos_binned[y_pos_out_field], 'ob')
    # pl.show()

    start_end_idx_in_field = [np.digitize([pos_t[idxs[0]], pos_t[idxs[-1]]], t) for idxs in idxs_per_in_field]
    start_end_idx_out_field = [np.digitize([pos_t[idxs[0]], pos_t[idxs[-1]]], t) for idxs in idxs_per_out_field]
    return start_end_idx_in_field, start_end_idx_out_field


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