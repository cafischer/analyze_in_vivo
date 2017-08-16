import numpy as np
import matplotlib.pyplot as pl
from load import load_VI
from cell_characteristics.analyze_APs import get_AP_onset_idxs


if __name__ == '__main__':

    data_dir = '../data/'
    # "10217003",  -> good  --> sag, doublet
    # "10n10000",  -> good  --> sag, doublet
    # "11303000",  -> pretty noisy --> no sag, bursts
    # "11910002",  -> good --> sag, doublet, big DAPs
    # "11n30004",  -> good --> sag, doublet, little DAPs
    # "11d13006"   -> good --> sag, doublet
    cell_id = "11n30004"
    v, t, i_inj = load_VI(data_dir, cell_id)
    dt = t[1] - t[0]
    start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
    end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]

    for i in range(np.shape(v)[0]):
        i_amp = np.round(i_inj[i, start_step+(end_step-start_step) / 2], 1)
        print i_amp

        # pl.figure()
        # pl.plot(t, i_inj[i, :])
        # pl.plot(t[start_step:end_step], i_inj[i, start_step:end_step], 'r')
        # pl.show()

        pl.figure()
        pl.plot(t, v[i, :], 'k')
        pl.show()

        AP_onsets = get_AP_onset_idxs(v[i, start_step:end_step], threshold=-10) + start_step
        if len(AP_onsets) <= 2:
            continue
        AP_onsets = np.concatenate((AP_onsets, np.array([end_step])))

        if (AP_onsets[1] - AP_onsets[0]) * dt < 15:
            first_spikes = v[i, AP_onsets[0]:AP_onsets[2]]
            other_spikes = []
            for s, e in zip(AP_onsets[2:-1], AP_onsets[3:]):
                if (e - s) * dt >= 15:
                    other_spikes.append(v[i, s:e])
        else:
            first_spikes = v[i, AP_onsets[0]:AP_onsets[1]]
            other_spikes = []
            for s, e in zip(AP_onsets[1:-1], AP_onsets[2:]):
                if (e - s) * dt >= 15:
                    other_spikes.append(v[i, s:e])

        pl.figure()
        pl.plot(np.arange(0, len(first_spikes)) * dt, first_spikes)
        pl.show()

        pl.figure()
        for spike in other_spikes:
            pl.plot(np.arange(0, len(spike)) * dt, spike)
        pl.show()