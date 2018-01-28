import numpy as np
import matplotlib.pyplot as pl
import os
from load import load_VI
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics.analyze_APs import get_fAHP_min_idx_using_splines, get_AP_max_idx
from cell_fitting.data import set_v_rest
from cell_characteristics import to_idx


def get_first_spikes(v_mat, start_step, end_step, AP_threshold, least_n_spikes=3):
    for i in range(np.shape(v_mat)[0]):
        AP_onset_idxs = get_AP_onset_idxs(v_mat[i, start_step:end_step], threshold=AP_threshold) + start_step
        if len(AP_onset_idxs) >= least_n_spikes:
            return v_mat[i, :], AP_onset_idxs


def split_first_and_other_spikes(v, dt, AP_onset_idxs, ISI_doublet, end_step, before_onset=None):

    before_onset_idx = 0 if before_onset is None else to_idx(before_onset, dt)

    AP_onset_idxs = np.concatenate((AP_onset_idxs, np.array([end_step])))

    if (AP_onset_idxs[1] - AP_onset_idxs[0]) * dt < ISI_doublet:
        first_spikes = v[AP_onset_idxs[0]-before_onset_idx:AP_onset_idxs[2]]
        other_spikes = []
        for s, e in zip(AP_onset_idxs[2:-1], AP_onset_idxs[3:]):
            if (e - s) * dt >= 15:
                other_spikes.append(v[s-before_onset_idx:e])
    else:
        first_spikes = v[AP_onset_idxs[0]-before_onset_idx:AP_onset_idxs[1]]
        other_spikes = []
        for s, e in zip(AP_onset_idxs[1:-1], AP_onset_idxs[2:]):
            if (e - s) * dt >= 15:
                other_spikes.append(v[s-before_onset_idx:e])

    return first_spikes, other_spikes


def make_spike_mat(spikes, start, end):
    spike_mat = np.zeros((len(spikes), end-start))
    for i, spike in enumerate(spikes):
        spike_mat[i, :] = spike[start:end]
    return spike_mat


def cut_at_fAHP_min(other_spikes, after_fAHP_idx, dt):
    other_spikes_mat = []
    for i, v_trace in enumerate(other_spikes):
        AP_max_idx = get_AP_max_idx(v_trace, 0, len(v_trace), interval=int(round(2 / dt)))

        std = np.std(v_trace[-int(round(2.0 / dt)):-int(round(1.0 / dt))])
        w = np.ones(len(v_trace)) / std
        fAHP_min_idx = get_fAHP_min_idx_using_splines(v_trace, np.arange(len(v_trace)) * dt, AP_max_idx, len(v_trace),
                                                      order=50, interval=int(round(4 / dt)), w=w)
        if fAHP_min_idx is None:
            continue
        other_spikes_mat.append(v_trace[fAHP_min_idx:fAHP_min_idx + after_fAHP_idx + 1])

        # t_trace = np.arange(len(v_trace)) * dt
        # pl.figure()
        # pl.plot(t_trace, v_trace)
        # pl.plot(t_trace[fAHP_min_idx], v_trace[fAHP_min_idx], 'o')
        # pl.show()
    return np.vstack(other_spikes_mat)


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape'
    data_dir = '../data/'
    # "10217003",  -> good  --> sag, doublet
    # "10n10000",  -> good  --> sag, doublet
    # "11910002",  -> good --> sag, doublet, big DAPs
    # "11n30004",  -> good --> sag, doublet, little DAPs
    # "11d13006"   -> good --> sag, doublet
    # "11303000",  -> pretty noisy --> no sag, bursts
    cell_ids = ["10217003", "10n10000", "11910002", "11n30004", "11d13006"]

    AP_threshold = -10  # mV
    ISI_doublet = 15  # ms

    all_first_spikes = []
    all_other_spikes = []
    for cell_id in cell_ids:
        v_mat, t, i_inj = load_VI(data_dir, cell_id)
        dt = t[1] - t[0]
        after_fAHP_idx = int(round(30 / dt))
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]

        v_trace, AP_onset_idxs = get_first_spikes(v_mat, start_step, end_step, AP_threshold)
        first_spikes, other_spikes = split_first_and_other_spikes(v_trace, dt, AP_onset_idxs, ISI_doublet, end_step)

        other_spikes_mat = cut_at_fAHP_min(other_spikes, after_fAHP_idx, dt)
        all_other_spikes.append(other_spikes_mat)

    all_other_spikes = np.concatenate(all_other_spikes)

    v_rest = -45
    all_other_spikes = set_v_rest(all_other_spikes, np.array([all_other_spikes[:, 0]]).T,
                                  np.ones((np.shape(all_other_spikes)[0], 1)) * v_rest)

    pl.figure()
    for spike in all_other_spikes:
        pl.plot(np.arange(0, len(spike)) * dt, spike)
    pl.show()
