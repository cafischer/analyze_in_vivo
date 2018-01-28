import numpy as np
import matplotlib.pyplot as pl
import os
from load import load_VI
from cell_characteristics.analyze_APs import get_AP_onset_idxs

if __name__ == '__main__':

    save_dir = './results/schmidthieber/resting_potential/traces'
    data_dir = './data/'
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006",
                "11d13006", "12213002"]

    for cell_idx, cell_id in enumerate(cell_ids):
        v, t, i_inj = load_VI(data_dir, cell_id)
        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj[0, :])) > 0.05)[0][0]

        save_dir_fig = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        for i in range(np.shape(v)[0]):
            i_amp = np.round(i_inj[i, start_step + (end_step - start_step) / 2], 2)
            if i_amp == 0.5:
                Rs = 80  # megaOhm = mV / nA
                v_rs = v[i, :] + i_inj[i, :] * Rs
                pl.figure()
                pl.plot(t, v[i, :], 'k')

                pl.plot(t, v_rs, 'b')
                pl.ylabel('Membrane potential (mV)', fontsize=16)
                pl.xlabel('Time (ms)', fontsize=16)
                pl.show()
                break

        # pl.figure()
        # for i in range(np.shape(i_inj)[0]):
        #     pl.plot(t, i_inj[i, :])
        # pl.ylabel('Current (nA)', fontsize=16)
        # pl.xlabel('Time (ms)', fontsize=16)
        # pl.show()