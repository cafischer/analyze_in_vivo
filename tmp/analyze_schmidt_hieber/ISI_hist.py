from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISIs, get_cumulative_ISI_hist, \
    plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells
from analyze_in_vivo.load.load_schmidt_hieber import load_full_runs
from scipy.stats import ks_2samp
from itertools import combinations
pl.style.use('paper')


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies

    save_dir = '../results/schmidthieber/full_traces/ISI_hist'
    data_dir = '../data/schmidt-hieber'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    cell_ids_plot = [cell_id.split('_')[1] for cell_id in cell_ids]

    # parameter
    max_ISI = 200
    bins = np.arange(0, max_ISI+2, 2.0)

    # over cells
    ISIs_per_cellid = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist_y = [0] * len(cell_ids)
    cum_ISI_hist_x = [0] * len(cell_ids)

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]

        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5

        # ISI histogram
        AP_max_idxs = get_AP_max_idxs(v, AP_threshold, dt, interval=2)
        ISIs = get_ISIs(AP_max_idxs, t)
        ISI_hist[i, :] = get_ISI_hist(ISIs, bins)

        # ISIs
        n_ISIs[i] = len(ISIs)
        ISIs_per_cellid[i] = ISIs

        cum_ISI_hist_y[i], cum_ISI_hist_x[i] = get_cumulative_ISI_hist(ISIs, upper_bound=None)

        # save and plot
        save_dir_cell = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # plot_cumulative_ISI_hist(cum_ISI_hist[i, :], bins_cum, save_dir_cell)
        # plot_ISI_hist(ISI_hist[i, :], bins, save_dir_cell)
        # pl.close()

    # plot all cumulative ISI histograms in one
    ISIs_all = np.array([item for sublist in ISIs_per_cellid for item in sublist])
    cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all, upper_bound=None)
    plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
                                       cell_ids_plot, max_ISI, save_dir)
    pl.show()

    # for each pair of cells two sample Kolmogorov Smironov test (Note: ISIs are cut at 200 ms (=max(bins)))
    save_dir_ks = os.path.join(save_dir, 'kolmogorov_smirnov_2cells')
    if not os.path.exists(save_dir_ks):
        os.makedirs(save_dir_ks)

    p_val_dict = {}
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        D, p_val = ks_2samp(ISIs_per_cellid[i1], ISIs_per_cellid[i2])
        p_val_dict[(i1, i2)] = p_val
        print 'p-value for cell '+str(cell_ids_plot[i1]) \
              + ' and cell '+str(cell_ids_plot[i2]) + ': %.3f' % p_val

        # pl.figure()
        # cm = pl.cm.get_cmap('plasma')
        # colors = cm(np.linspace(0, 1, len(cell_ids)))
        # pl.title(str(D)) #pl.title('p-value: %.3f' % p_val)
        # pl.plot(bins_cum[:-1], cum_ISI_hist[i1, :], label=cell_ids[i1].split('_')[1], color=colors[i1])
        # pl.plot(bins_cum[:-1], cum_ISI_hist[i2, :], label=cell_ids[i2].split('_')[1], color=colors[i2])
        # max_diff_idx = np.argmax(np.abs(cum_ISI_hist[i1, :] - cum_ISI_hist[i2, :]))
        # pl.annotate('', xy=(bins_cum[max_diff_idx], cum_ISI_hist[i1, max_diff_idx]),
        #                     xytext=(bins_cum[max_diff_idx], cum_ISI_hist[i2, max_diff_idx]),
        #                     arrowprops=dict(arrowstyle="<->", color='k'))
        # pl.xlabel('ISI (ms)')
        # pl.ylabel('Count')
        # pl.legend(fontsize=16, loc='lower right')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_ks,
        #                         'cum_ISI_'+str(cell_ids[i1].split('_')[1])+'_'+str(cell_ids[i2].split('_')[1])+'.png'))
        # pl.show()

    plot_cumulative_comparison_all_cells(cum_ISI_hist_x, cum_ISI_hist_y, cell_ids_plot, p_val_dict,
                                         os.path.join(save_dir_ks, 'comparison_cum_ISI.png'))
    pl.show()