from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISI_hist, plot_ISI_hist, get_cumulative_ISI_hist, plot_cumulative_ISI_hist
from load import load_full_runs
from scipy.stats import ks_2samp
from itertools import combinations
pl.style.use('paper')


def plot_cum_ISI_all_cells(cum_ISI_hist, bins_cum, cell_ids, save_dir):
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    pl.figure()
    for i, cell_id in enumerate(cell_ids):
        pl.plot(bins_cum[:-1], cum_ISI_hist[i, :], label=cell_id.split('_')[1], color=colors[i])
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    pl.legend(fontsize=16, loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'cum_ISI_hist.png'))


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

    # over cells
    ISIs_per_cellid = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist = np.zeros((len(cell_ids), len(bins_cum)-1))

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]

        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5

        # ISI histogram
        ISI_hist[i, :], ISIs = get_ISI_hist(v, t, AP_threshold, bins)
        n_ISIs[i] = len(ISIs)
        ISIs_per_cellid[i] = ISIs[ISIs <= bins_cum[-1]]
        cum_ISI_hist[i, :], _ = get_cumulative_ISI_hist(v, t, AP_threshold, bins=bins_cum)

        # save and plot
        save_dir_cell = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # plot_cumulative_ISI_hist(cum_ISI_hist[i, :], bins_cum, save_dir_cell)
        # plot_ISI_hist(ISI_hist[i, :], bins, save_dir_cell)
        # pl.close()

    # plot all cumulative ISI histograms in one
    plot_cum_ISI_all_cells(cum_ISI_hist, bins_cum, cell_ids, save_dir)

    # for each pair of cells two sample Kolmogorov Smironov test (Note: ISIs are cut at 200 ms (=max(bins)))
    save_dir_ks = os.path.join(save_dir, 'kolmogorov_smirnov_2cells')
    if not os.path.exists(save_dir_ks):
        os.makedirs(save_dir_ks)

    p_val_dict = {}
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        D, p_val = ks_2samp(ISIs_per_cellid[i1], ISIs_per_cellid[i2])
        p_val_dict[(i1, i2)] = p_val
        print 'p-value for cell '+str(cell_ids[i1].split('_')[1]) \
              + ' and cell '+str(cell_ids[i2].split('_')[1]) + ': %.3f' % p_val

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

    fig, ax = pl.subplots(len(cell_ids), len(cell_ids), sharex='all', sharey='all', figsize=(10, 10))
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        ax[i2, i1].set_title('p-value: %.2f' % p_val_dict[(i1, i2)])
        ax[i2, i1].plot(bins_cum[:-1], cum_ISI_hist[i1, :], label=cell_ids[i1].split('_')[1], color=colors[i1])
        ax[i2, i1].plot(bins_cum[:-1], cum_ISI_hist[i2, :], label=cell_ids[i2].split('_')[1], color=colors[i2])
        max_diff_idx = np.argmax(np.abs(cum_ISI_hist[i1, :]-cum_ISI_hist[i2, :]))
        ax[i2, i1].annotate('', xy=(bins_cum[max_diff_idx], cum_ISI_hist[i1, max_diff_idx]),
                            xytext=(bins_cum[max_diff_idx], cum_ISI_hist[i2, max_diff_idx]),
                            arrowprops=dict(arrowstyle="<->", color='k'))
    # fig.text(0.06, 0.5, 'CDF', va='center', rotation='vertical', fontsize=18)
    # fig.text(0.5, 0.06, 'ISI (ms)', ha='center', fontsize=18)
    for i1 in range(len(cell_ids)):
        for i2 in range(len(cell_ids)):
            ax[i2, i1].set(xlabel=cell_ids[i1].split('_')[1], ylabel=cell_ids[i2].split('_')[1])
            ax[i2, i1].label_outer()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_ks, 'comparison_cum_ISI.png'))
    pl.show()