from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs
from load import load_full_runs
pl.style.use('paper')


def get_ISI_hist_and_cum_hist(ISIs, bins, bins_cum):
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)
    ISI_hist_tmp, bin_edges = np.histogram(ISIs, bins=bins_cum)
    ISI_hist_tmp = ISI_hist_tmp / np.sum(ISI_hist_tmp)
    cum_ISI_hist = np.cumsum(ISI_hist_tmp)
    return ISI_hist, cum_ISI_hist


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies

    save_dir = '../results/schmidthieber/full_traces/ISI_hist'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]

    # parameter
    bins = np.arange(0, 200+2, 2.0)
    bins_cum = np.arange(0, 200+0.1, 0.1)
    short_ISI = 4  # ms

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist = np.zeros((len(cell_ids), len(bins_cum)-1))

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]

        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5

        # ISIs
        ISIs = get_ISIs(v, t, AP_threshold)

        where_short_ISI = np.where(ISIs < short_ISI)[0]
        before_short_ISI = ISIs[where_short_ISI-1]
        after_short_ISI = ISIs[where_short_ISI+1]

        ISI_hist_before, cum_ISI_hist_before = get_ISI_hist_and_cum_hist(before_short_ISI, bins, bins_cum)
        ISI_hist_after, cum_ISI_hist_after = get_ISI_hist_and_cum_hist(after_short_ISI, bins, bins_cum)

        # save and plot
        save_dir_cell = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        fig, ax = pl.subplots(2, 2, sharex='all', figsize=(7.4, 5.8))
        ax[0, 0].bar(bins[:-1], ISI_hist_before, bins[1] - bins[0], color='0.5')
        ax[0, 1].bar(bins[:-1], ISI_hist_after, bins[1] - bins[0], color='0.5')
        ax[1, 0].plot(bins_cum[:-1], cum_ISI_hist_before, color='0.5')
        ax[1, 1].plot(bins_cum[:-1], cum_ISI_hist_after, color='0.5')
        ax[0, 0].set_title('Before short ISI', fontsize=18)
        ax[0, 1].set_title('After short ISI', fontsize=18)
        ax[1, 0].set_xlabel('ISI (ms)')
        ax[1, 1].set_xlabel('ISI (ms)')
        ax[0, 0].set_ylabel('Count')
        ax[1, 0].set_ylabel('CDF')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'before_after_short_ISI.png'))
        #pl.show()
        pl.close()