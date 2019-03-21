from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.spike_events import get_burst_lengths_and_n_single
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
    ISIs_cells = load_ISIs()

    burst_ISI = 8  # ms
    bins = np.arange(1, 15 + 1, 1)

    folder = 'burst_ISI_' + str(burst_ISI)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    count_spikes = np.zeros((len(ISIs_cells), len(bins)-1))
    fraction_single = np.zeros(len(ISIs_cells))

    for cell_idx in range(len(ISIs_cells)):
        print cell_idx
        ISIs = ISIs_cells[cell_idx]

        short_ISI_indicator = np.concatenate((ISIs <= burst_ISI, np.array([False])))
        burst_lengths, n_single = get_burst_lengths_and_n_single(short_ISI_indicator)
        count_spikes[cell_idx, :] = np.histogram(burst_lengths, bins)[0]
        count_spikes[cell_idx, 0] = n_single
        fraction_single[cell_idx] = count_spikes[cell_idx, 0] / np.sum(count_spikes[cell_idx, :])

        # # plot
        # pl.figure()
        # pl.bar(bins[:-1], count_spikes[cell_idx, :], width=0.7, color='0.5', align='center')
        # pl.xlabel('# Spikes in event')
        # pl.ylabel('Frequency')
        # pl.show()
        # pl.close('all')

    # save
    np.save(os.path.join(save_dir_img, 'fraction_single.npy'), fraction_single)