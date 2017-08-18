from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from load import load_field_crossings, get_stellate_info
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISI_hists_into_outof_field, plot_ISI_hist, \
    plot_ISI_hist_into_outof_field


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/ISI_hist'
    save_dir_data = '../results/' + folder + '/data'

    # parameter
    AP_thresholds = [-40, -40, -50, -30, -40, -50]
    bins = np.arange(0, 200, 2)

    # over cells
    file_names = os.listdir(save_dir_data)
    n_ISIs = [0] * len(file_names)
    ISI_hist = np.zeros((len(file_names), len(bins)-1))
    ISI_hist_into = np.zeros((len(file_names), len(bins) - 1))
    ISI_hist_outof = np.zeros((len(file_names), len(bins) - 1))

    # over all field crossings
    for i, file_name in enumerate(file_names):

        # load
        v = np.load(os.path.join(save_dir_data, file_name, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        # pl.figure()
        # pl.plot(t, v)
        # pl.show()

        AP_threshold = np.max(v) - np.abs((np.min(v) - np.max(v)) / 3)

        # ISI histogram
        ISI_hist[i, :], ISIs = get_ISI_hist(v, t, AP_threshold, bins)
        n_ISIs[i] = len(ISIs)

        # in and out field ISIs
        field_pos_idxs = [int(round(len(v) / 2))]
        ISI_hist_into[i, :], ISI_hist_outof[i, :] = get_ISI_hists_into_outof_field(v, t, AP_threshold, bins,
                                                                                   field_pos_idxs)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        plot_ISI_hist(ISI_hist[i, :], bins, save_dir_cell_field_crossing)
        plot_ISI_hist_into_outof_field(ISI_hist_into[i, :], ISI_hist_outof[i, :], bins, save_dir_cell_field_crossing)

    # save and plots
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ISI_hist = np.sum(ISI_hist, 0)
    ISI_hist_into = np.sum(ISI_hist_into, 0)
    ISI_hist_outof = np.sum(ISI_hist_outof, 0)
    n_ISIs = np.sum(n_ISIs)

    n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
    percent_doublets = n_doublets / n_ISIs
    print('Percent doublets: ', percent_doublets)

    n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
    percent_theta = n_theta / n_ISIs
    print('Percent theta: ', percent_theta)

    plot_ISI_hist(ISI_hist, bins, save_dir)
    plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir)