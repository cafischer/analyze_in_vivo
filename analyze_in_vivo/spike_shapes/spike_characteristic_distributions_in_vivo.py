import numpy as np
import matplotlib.pyplot as pl
import os
from cell_fitting.data import set_v_rest
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.load import load_VI


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape'
    data_dir = '../data/'
    #cell_ids = ["10217003", "10n10000", "11910002", "11n30004", "11d13006"]
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006", "11d13006", "12213002"]
    protocol = 'IV'

    ISI_doublet = 15  # ms
    v_rest = -60

    all_spikes = []
    for cell_id in cell_ids:
        v_mat, t, i_inj = load_VI(data_dir, cell_id)

        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        start_step += to_idx(3, dt)  # to cut off transient in the beginning
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]
        after_onset = to_idx(30, dt)
        before_onset = to_idx(5, dt)

        v_mat_step = np.zeros((len(v_mat), end_step-start_step))
        for i in range(len(v_mat)):
            v_mat_step[i, :] = set_v_rest(v_mat[i, start_step:end_step], np.mean(v_mat[i, start_step:end_step]), v_rest)

        spikes_cell = []
        for v in v_mat_step:
            AP_threshold = np.min(v) + np.abs(np.max(v) - np.min(v)) * (2./3)
            print AP_threshold
            onsets = get_AP_onset_idxs(v, AP_threshold)
            for onset in onsets:
                if onset - before_onset >= 0 and onset + after_onset <= len(v):
                    spikes_cell.append(v[onset - before_onset:onset + after_onset])
        spikes_cell = np.vstack(spikes_cell)

        pl.figure()
        for v in v_mat:
            pl.plot(np.arange(len(v))*dt, v)
        pl.show()

        pl.figure()
        for v in spikes_cell:
            pl.plot(np.arange(len(v))*dt, v)
        pl.show()

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