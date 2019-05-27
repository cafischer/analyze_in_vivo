from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_cell_ids_burstgroups
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from analyze_in_vivo.analyze_domnisoru.autocorr import *
from grid_cell_stimuli.ISI_hist import get_ISIs
import pandas as pd
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, \
    get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.isi import get_ISI_hist_peak_and_width
from scipy.signal import argrelmax
pl.style.use('paper')


if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr/kde'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_ids = load_cell_ids(save_dir, 'grid_cells')
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    max_lag = 50
    sigma_smooth = 1  # ms  has to be given
    dt_kde = 0.05  # ms (same as dt data as lower bound for precision)
    t_kde = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
    save_dir_img = os.path.join(save_dir_img)

    folder = 'max_lag_' + str(max_lag) + '_sigma_smooth_'+str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    autocorr_kde_cells = np.zeros((len(cell_ids), int(2 * max_lag / dt_kde + 1)))
    peak_autocorr = np.zeros(len(cell_ids))
    width_autocorr = np.zeros(len(cell_ids))
    peak_NB = np.zeros(len(get_cell_ids_burstgroups()['NB']))
    j = 0

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # compute KDE
        ISIs = get_ISIs(AP_max_idxs, t)
        SIs = get_all_SIs_lower_max_lag_except_zero(ISIs, max_lag)
        kde = perform_kde(SIs, sigma_smooth)
        autocorr_kde_cells[cell_idx, :] = evaluate_kde(t_kde, kde)

        # get peak of autocorr
        max_lag_idx = to_idx(max_lag, dt_kde)
        #peak_autocorr[cell_idx] = t_kde[max_lag_idx:][np.argmax(autocorr_kde_cells[cell_idx, max_lag_idx:])]
        peak_autocorr[cell_idx], width_autocorr[cell_idx] = get_ISI_hist_peak_and_width(autocorr_kde_cells[cell_idx, max_lag_idx:],
                                                                                        t_kde[max_lag_idx:])

        if cell_id in get_cell_ids_burstgroups()['NB']:
            peak_idx = argrelmax(autocorr_kde_cells[cell_idx, max_lag_idx:])[0][0]
            peak_NB[j] = t_kde[max_lag_idx + peak_idx]
            #peak_y = autocorr_kde_cells[cell_idx, max_lag_idx + peak_idx]

            # pl.figure()
            # pl.plot(t_kde, autocorr_kde_cells[cell_idx, :], color='k')
            # pl.plot(peak_NB[j], peak_y, 'or')
            # pl.xlabel('Time (ms)')
            # pl.ylabel('Spike-time autocorrelation')
            # pl.xlim(-max_lag, max_lag)
            # pl.ylim(0, None)
            # pl.tight_layout()
            # pl.show()

            j += 1

    #
    print 'mean peak NB: %.2f' % np.mean(peak_NB)
    print 'std peak NB: %.2f' % np.std(peak_NB)

    # save autocorrelation
    np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_kde_cells)
    np.save(os.path.join(save_dir_img, 'peak_autocorr.npy'), peak_autocorr)
    np.save(os.path.join(save_dir_img, 'width_autocorr.npy'), width_autocorr)

    # plots
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'
    # params = {'legend.fontsize': 9}
    # pl.rcParams.update(params)
    #
    # plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
    #                    max_lag=max_lag, kernel_cells=kde_cells)
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation_with_kde, plot_kwargs,
    #                         xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
    #                         save_dir_img=os.path.join(save_dir_img, 'autocorr.png'))

    # plot peak autocorr
    cell_type_dict = get_celltype_dict(save_dir)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    cell_ids = np.array(cell_ids)
    fig, ax = pl.subplots(1, 1, figsize=(4, 5))
    plot_with_markers(ax, np.zeros(np.sum(burst_label)), peak_autocorr[burst_label],
                      cell_ids[burst_label], cell_type_dict,
                      edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(ax, np.ones(np.sum(~burst_label)), peak_autocorr[~burst_label],
                                cell_ids[~burst_label], cell_type_dict,
                                edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Bursty', 'Non-bursty'])
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylabel('Peak of the spike-time autocorrelation')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'peak_autocorr.png'))
    pl.show()
