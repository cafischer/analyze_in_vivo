import numpy as np
import matplotlib.pyplot as pl
import os
from spike_shape import get_first_spikes, split_first_and_other_spikes, make_spike_mat, cut_at_fAHP_min
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj, set_v_rest
from cell_fitting.DAP_population import get_spike_characteristics


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    file_names = os.listdir(data_dir)
    cell_ids = [f_n[:-4] for f_n in file_names]
    #cell_ids = ["2015_08_25b", "2015_08_25h", "2015_08_27d", "2015_08_26b", "2015_08_26f"]
    protocol = 'IV'

    AP_threshold = -10  # mV
    ISI_doublet = 15  # ms
    v_rest = -60

    all_spikes = []
    for cell_id in cell_ids:
        try:
            v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        except KeyError:
            continue
        t = t_mat[0, :]
        i_inj = get_i_inj(protocol, sweep_idxs)

        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]
        spike_len = int(round(30 / dt))

        first_spikes_there = get_first_spikes(v_mat, start_step, end_step, AP_threshold)
        if first_spikes_there is None:
            continue
        v_trace, AP_onset_idxs = first_spikes_there
        first_spikes, other_spikes = split_first_and_other_spikes(v_trace, dt, AP_onset_idxs, ISI_doublet, end_step)
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
    AP_interval = int(round(3 / dt))
    std_idxs = (-int(round(2 / dt)), -int(round(1 / dt)))
    DAP_interval = int(round(5 / dt))
    order_fAHP_min = int(round(0.3 / dt))  # how many points to consider for the minimum
    order_DAP_max = int(round(2.0 / dt))  # how many points to consider for the minimum
    dist_to_DAP_max = int(round(1 / dt))


    AP_matrix = np.vstack(all_spikes)
    t_window = np.arange(np.shape(AP_matrix)[1]) * dt
    AP_amp, AP_width, DAP_amp, DAP_deflection, DAP_width, DAP_time, DAP_lin_slope, DAP_exp_slope = \
        get_spike_characteristics(AP_matrix, t_window, AP_interval, std_idxs, DAP_interval, np.min(AP_matrix, 1),
                                  order_fAHP_min, order_DAP_max, dist_to_DAP_max, check=False)
    spike_characteristics_mat = np.vstack((AP_amp, AP_width, DAP_amp, DAP_deflection, DAP_width, DAP_time, DAP_lin_slope,
                                          DAP_exp_slope)).T
    not_nan = np.logical_not(np.any(np.isnan(spike_characteristics_mat), 1))
    spike_characteristics_mat = spike_characteristics_mat[not_nan, :]
    spike_characteristics_names = ['AP_amp', 'AP_width', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time',
                                   'DAP_lin_slope', 'DAP_exp_slope']
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

    np.save(os.path.join(save_dir, 'spike_characteristics_names.npy'), spike_characteristics_names)
    np.save(os.path.join(save_dir, 'spike_characteristics_mat_vitro.npy'), spike_characteristics_mat)
    np.save(os.path.join(save_dir, 'AP_mat_vitro.npy'), AP_matrix)
    np.save(os.path.join(save_dir, 't_vitro.npy'), t_window)