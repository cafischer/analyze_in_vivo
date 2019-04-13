import numpy as np
import matplotlib.pyplot as pl
import os
from spike_shape import get_first_spikes, split_first_and_other_spikes, make_spike_mat, cut_at_fAHP_min
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from cell_fitting.data import set_v_rest
from cell_fitting.DAP_population import get_spike_characteristics


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_ids = ["2015_08_25b", "2015_08_25h", "2015_08_27d", "2015_08_26b", "2015_08_26f"]
    protocol = 'IV'

    AP_threshold = -10  # mV
    ISI_doublet = 15  # ms

    all_other_spikes = []
    for cell_id in cell_ids:
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        i_inj = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])
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
                              np.ones((np.shape(all_other_spikes)[0], 1))*v_rest)

pl.figure()
for spike in all_other_spikes:
    pl.plot(np.arange(0, len(spike)) * dt, spike)
pl.show()