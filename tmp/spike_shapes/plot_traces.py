import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load import load_VI
from cell_characteristics.analyze_APs import get_AP_onset_idxs

if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape/traces'
    data_dir = '../data/'
    # "10217003",  -> good  --> sag, doublet
    # "10n10000",  -> good  --> sag, doublet
    # "11910002",  -> good --> sag, doublet, big DAPs
    # "11n30004",  -> good --> sag, doublet, little DAPs
    # "11d13006"   -> good --> sag, doublet
    # "11303000",  -> pretty noisy --> no sag, bursts
    #cell_ids = ["10217003", "10n10000", "11910002", "11n30004", "11d13006"]
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006", "11d13006", "12213002"]

    for cell_id in cell_ids:
        print cell_id
        v, t, i_inj = load_VI(data_dir, cell_id)
        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]

        save_dir_fig = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        for i in range(np.shape(v)[0]):
            i_amp = np.round(i_inj[i, start_step + (end_step - start_step) / 2], 1)

            # if i_amp == -0.1:
            #     pl.figure()
            #     pl.plot(t, v[i, :], 'k')
            #     pl.ylabel('Membrane potential (mV)', fontsize=16)
            #     pl.xlabel('Time (ms)', fontsize=16)
            #     pl.savefig(os.path.join(save_dir_fig, 'sag.svg'))
            #     pl.show()
            #
            # AP_onsets = get_AP_onset_idxs(v[i, start_step:end_step], threshold=-10) + start_step
            # if len(AP_onsets) <= 2:
            #     continue

            pl.figure()
            pl.plot(t, v[i, :], 'k')
            pl.ylabel('Membrane potential (mV)', fontsize=16)
            pl.xlabel('Time (ms)', fontsize=16)
            #pl.savefig(os.path.join(save_dir_fig, 'first_spikes.svg'))
            pl.show()

            #break
