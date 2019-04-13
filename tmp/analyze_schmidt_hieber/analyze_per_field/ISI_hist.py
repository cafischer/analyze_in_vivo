from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISI_hists_into_outof_field, plot_ISI_hist, \
    plot_ISI_hist_into_outof_field, get_cumulative_ISI_hist, plot_cumulative_ISI_hist, \
    plot_cumulative_ISI_hist_into_outof
from itertools import combinations
from scipy.stats import ks_2samp


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/ISI_hist'
    save_dir_data = '../results/' + folder + '/data'

    # parameter
    bins = np.arange(0, 200+2, 2.0)
    bins_cum = np.arange(0, 200+0.1, 0.1)

    # over cells
    file_names = os.listdir(save_dir_data)
    n_ISIs = [0] * len(file_names)
    ISIs_per_field = [0] * len(file_names)
    ISI_hist = np.zeros((len(file_names), len(bins)-1))
    cum_ISI_hist = np.zeros((len(file_names), len(bins_cum)-1))
    ISI_hist_into = np.zeros((len(file_names), len(bins)-1))
    ISI_hist_outof = np.zeros((len(file_names), len(bins)-1))
    cum_ISI_hist_into = np.zeros((len(file_names), len(bins_cum)-1))
    cum_ISI_hist_outof = np.zeros((len(file_names), len(bins_cum) - 1))

    # over all field crossings
    for i, file_name in enumerate(file_names):
        # load
        v = np.load(os.path.join(save_dir_data, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        dt = t[1] - t[0]
        print dt
        # pl.figure()
        # pl.plot(t, v)
        # pl.show()

        AP_threshold = np.max(v) - np.abs((np.min(v) - np.max(v)) / 3)

        # ISI histogram
        ISI_hist[i, :], ISIs = get_ISI_hist(v, t, AP_threshold, bins)
        n_ISIs[i] = len(ISIs)
        ISIs_per_field[i] = ISIs[ISIs<=bins_cum[-1]]
        cum_ISI_hist[i, :], _ = get_cumulative_ISI_hist(v, t, AP_threshold, bins=bins_cum)

        # in and out field ISIs
        field_pos_idxs = [int(round(len(v) / 2))]
        ISI_hist_into[i, :], ISI_hist_outof[i, :] = get_ISI_hists_into_outof_field(v, t, AP_threshold, bins,
                                                                                   field_pos_idxs)
        ISI_hist_into_tmp, ISI_hist_outof_tmp = get_ISI_hists_into_outof_field(v, t, AP_threshold, bins_cum,
                                                                                   field_pos_idxs)
        cum_ISI_hist_into[i, :] = np.cumsum(ISI_hist_into_tmp)
        cum_ISI_hist_outof[i, :] = np.cumsum(ISI_hist_outof_tmp)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        # plot_cumulative_ISI_hist(cum_ISI_hist[i, :], bins_cum, save_dir_cell_field_crossing)
        # plot_ISI_hist(ISI_hist[i, :], bins, save_dir_cell_field_crossing)
        # plot_ISI_hist_into_outof_field(ISI_hist_into[i, :], ISI_hist_outof[i, :], bins, save_dir_cell_field_crossing)
        # plot_cumulative_ISI_hist_into_outof(cum_ISI_hist_into[i, :], cum_ISI_hist_outof[i, :], bins_cum,
        #                                     save_dir_cell_field_crossing)
        # pl.show()
        # pl.close()

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #
    # ISI_hist = np.sum(ISI_hist, 0)
    # ISI_hist_into = np.sum(ISI_hist_into, 0)
    # ISI_hist_outof = np.sum(ISI_hist_outof, 0)
    # n_ISIs = np.sum(n_ISIs)
    #
    # n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
    # percent_doublets = n_doublets / n_ISIs
    # print('Percent doublets: ', percent_doublets)
    #
    # n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
    # percent_theta = n_theta / n_ISIs
    # print('Percent theta: ', percent_theta)
    #
    # plot_ISI_hist(ISI_hist, bins, save_dir)
    # plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir)


    # compare ISI variability between fields for each cell
    save_dir_ks = os.path.join(save_dir, 'kolmogorov_smirnov_2cells')
    if not os.path.exists(save_dir_ks):
        os.makedirs(save_dir_ks)

    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    for i, cell_id in enumerate(cell_ids):
        fields_of_cell = np.array([cell_id in f_n for f_n in file_names])
        ISIs_per_field_for_cell = np.array(ISIs_per_field)[fields_of_cell]
        cum_ISI_hist_per_cell = cum_ISI_hist[fields_of_cell, :]
        n_fields = len(ISIs_per_field_for_cell)
        p_val_dict = {}
        for i1, i2 in combinations(range(n_fields), 2):
            D, p_val = ks_2samp(ISIs_per_field_for_cell[i1], ISIs_per_field_for_cell[i2])
            p_val_dict[(i1, i2)] = p_val
            print 'p-value for cell ' + str(cell_id.split('_')[1]) + ' field ' + str(i1+1) \
                  + ' and ' + str(i2+1) + ': %.3f' % p_val

            # pl.figure()
            # cm = pl.cm.get_cmap('plasma')
            # colors = cm(np.linspace(0, 1, len(cell_ids)))
            # pl.title(D) # pl.title('cell '+cell_id.split('_')[1]+', p-value: %.3f' % p_val)
            # pl.plot(bins_cum[:-1], cum_ISI_hist_per_cell[i1, :], label=str(i1+1), color=colors[i1])
            # pl.plot(bins_cum[:-1], cum_ISI_hist_per_cell[i2, :], label=str(i2+1), color=colors[i2])
            # max_diff_idx = np.argmax(np.abs(cum_ISI_hist_per_cell[i1, :] - cum_ISI_hist_per_cell[i2, :]))
            # pl.annotate('', xy=(bins_cum[max_diff_idx], cum_ISI_hist_per_cell[i1, max_diff_idx]),
            #                     xytext=(bins_cum[max_diff_idx], cum_ISI_hist_per_cell[i2, max_diff_idx]),
            #                     arrowprops=dict(arrowstyle="<->", color='k'))
            # pl.xlabel('ISI (ms)')
            # pl.ylabel('Count')
            # pl.legend(fontsize=16, loc='lower right')
            # pl.tight_layout()
            # pl.show()

        if n_fields >= 2:
            fig, ax=pl.subplots(n_fields, n_fields, sharex='all', sharey='all', figsize=(8, 8))
            cm = pl.cm.get_cmap('plasma')
            colors = cm(np.linspace(0, 1, n_fields))
            for i1, i2 in combinations(range(n_fields), 2):
                ax[i2, i1].set_title('p-value: %.2f' % p_val_dict[(i1, i2)])
                ax[i2, i1].plot(bins_cum[:-1], cum_ISI_hist_per_cell[i1, :], label=cell_ids[i1].split('_')[1], color=colors[i1])
                ax[i2, i1].plot(bins_cum[:-1], cum_ISI_hist_per_cell[i2, :], label=cell_ids[i2].split('_')[1], color=colors[i2])
                max_diff_idx = np.argmax(np.abs(cum_ISI_hist_per_cell[i1, :] - cum_ISI_hist_per_cell[i2, :]))
                ax[i2, i1].annotate('', xy=(bins_cum[max_diff_idx], cum_ISI_hist_per_cell[i1, max_diff_idx]),
                                    xytext=(bins_cum[max_diff_idx], cum_ISI_hist_per_cell[i2, max_diff_idx]),
                                    arrowprops=dict(arrowstyle="<->", color='k'))

            # fig.text(0.06, 0.5, 'CDF', va='center', rotation='vertical', fontsize=18)
            # fig.text(0.5, 0.06, 'ISI (ms)', ha='center', fontsize=18)
            for i1 in range(n_fields):
                for i2 in range(n_fields):
                    ax[i2, i1].set(xlabel=str(i1+1), ylabel=str(i2+1))
                    ax[i2, i1].label_outer()
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_ks, 'comparison_cum_ISI_'+cell_id+'.png'))
    pl.show()