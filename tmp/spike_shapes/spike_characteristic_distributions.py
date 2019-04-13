import numpy as np
import matplotlib.pyplot as pl
import os
from spike_shape import get_first_spikes, split_first_and_other_spikes
from cell_fitting.data import set_v_rest
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_characteristics import to_idx
from tmp.load import load_VI


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape'
    data_dir = '../data/'
    #cell_ids = ["10217003", "10n10000", "11910002", "11n30004", "11d13006"]
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006", "11d13006", "12213002"]
    protocol = 'IV'

    AP_threshold = -30  # mV
    ISI_doublet = 15  # ms
    v_rest = -60  # TODO

    all_spikes = []
    for cell_id in cell_ids:
        v_mat, t, i_inj = load_VI(data_dir, cell_id)

        pl.figure()
        for v in v_mat:
            pl.plot(t, v)
        pl.show()

        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]
        spike_len = int(round(30 / dt))

        first_spikes_there = get_first_spikes(v_mat, start_step, end_step, AP_threshold)
        if first_spikes_there is None:
            continue
        v_trace, AP_onset_idxs = first_spikes_there
        first_spikes, other_spikes = split_first_and_other_spikes(v_trace, dt, AP_onset_idxs, ISI_doublet, end_step,
                                                                  before_onset=5)
        other_spikes_tmp = []
        for spike in other_spikes:
            if len(spike) >= spike_len:
                other_spikes_tmp.append(spike[:spike_len])
        if len(other_spikes_tmp) == 0:
            continue
        other_spikes = np.vstack(other_spikes_tmp)
        other_spikes = set_v_rest(other_spikes, np.array([other_spikes[:, -1]]).T,
                                  np.ones((np.shape(other_spikes)[0], 1)) * v_rest)
        all_spikes.append(other_spikes)

    # get spike characteristics
    spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
    std_idx_times = (0, 2)
    spike_characteristics_names = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time']

    AP_matrix = np.vstack(all_spikes)
    t_window = np.arange(np.shape(AP_matrix)[1]) * dt
    characteristics = []
    for v in AP_matrix:
        v_rest = np.mean(v[to_idx(0, dt):to_idx(5, dt)])  # TODO
        v_rest = v[-1]
        characteristics.append(
            get_spike_characteristics(v, t_window, spike_characteristics_names,
                                      v_rest=v_rest, std_idx_times=std_idx_times, check=True,
                                      **spike_characteristics_dict))  # TODO
    spike_characteristics_mat = np.array(np.vstack((characteristics)), dtype=float)
    not_nan = np.logical_not(np.any(np.isnan(spike_characteristics_mat), 1))
    spike_characteristics_mat = spike_characteristics_mat[not_nan, :]
    AP_matrix = AP_matrix[not_nan, :]

    # plot AP_matrix
    # pl.figure()
    # for spike in AP_matrix:
    #     pl.plot(np.arange(0, len(spike)) * dt, spike)
    # pl.show()

    # plot distributions
    for i in range(np.shape(spike_characteristics_mat)[1]-2):
        pl.figure()
        pl.title(spike_characteristics_names[i])
        pl.hist(spike_characteristics_mat[:, i])
        pl.show()

    np.save(os.path.join(save_dir, 'spike_characteristics_mat.npy'), spike_characteristics_mat)
    np.save(os.path.join(save_dir, 'AP_mat.npy'), AP_matrix)
    np.save(os.path.join(save_dir, 't.npy'), t_window)