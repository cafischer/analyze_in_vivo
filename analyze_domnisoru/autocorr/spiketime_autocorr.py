from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli import get_AP_max_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train, smooth_firing_rate
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from analyze_in_vivo.analyze_domnisoru.autocorr import *
from grid_cell_stimuli.ISI_hist import get_ISIs
import pandas as pd
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, \
    get_cell_ids_DAP_cells
pl.style.use('paper_subplots')


def plot_autocorrelation(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    ax.set_xlim(-max_lag, max_lag)


def plot_autocorrelation_with_kde(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag, kernel_cells, dt_kernel=0.01):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    t_kernel = np.arange(-max_lag, max_lag+dt_kernel, dt_kernel)
    ax.plot(t_kernel, kernel_cells[cell_idx].pdf(t_kernel), color='k')
    ax.set_xlim(-max_lag, max_lag)


if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    bin_width = 1  # ms
    max_lag = 50
    sigma_smooth = None  # ms  None for no smoothing
    dt_kde = 0.05  # ms (same as dt data as lower bound for precision)
    t_kde = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
    save_dir_img = os.path.join(save_dir_img)

    folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_'+str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    autocorr_cells = np.zeros((len(cell_ids), int(2 * max_lag / bin_width + 1)))
    autocorr_kde_cells = np.zeros((len(cell_ids), int(2 * max_lag / dt_kde + 1)))
    kde_cells = np.zeros(len(cell_ids), dtype=object)
    peak_autocorr = np.zeros(len(cell_ids))
    SIs_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # get ISIs
        ISIs = get_ISIs(AP_max_idxs, t)

        # get autocorrelation
        autocorr_cells[cell_idx, :], t_autocorr, bins = get_autocorrelation_by_ISIs(ISIs, max_lag=max_lag,
                                                                                    bin_width=bin_width)

        # compute KDE
        if sigma_smooth is not None:
            SIs = get_all_SIs_lower_max_lag_except_zero(ISIs, max_lag)
            kde_cells[cell_idx] = perform_kde(SIs, sigma_smooth)
            autocorr_kde_cells[cell_idx, :] = evaluate_kde(t_kde, kde_cells[cell_idx])

        # get peak of autocorr
        if sigma_smooth is not None:
            max_lag_idx = to_idx(max_lag, dt_kde)
            peak_autocorr[cell_idx] = t_kde[max_lag_idx:][np.argmax(autocorr_kde_cells[cell_idx, max_lag_idx:])]
        else:
            max_lag_idx = to_idx(max_lag, bin_width)
            peak_autocorr[cell_idx] = t_autocorr[max_lag_idx:][np.argmax(autocorr_cells[cell_idx, max_lag_idx:])]

        # pl.figure()
        # pl.bar(bins[:-1], autocorr_cells[cell_idx, :], bin_width, color='r', align='center', alpha=0.5)
        # t_kernel = np.arange(-50, 50+01, 0.1)
        # pl.plot(t_kernel, kernel.pdf(t_kernel), color='k')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # pl.show()

    # save autocorrelation
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_kde_cells)
    else:
        np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_cells)
    np.save(os.path.join(save_dir_img, 'peak_autocorr.npy'), peak_autocorr)

    # plots
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    if sigma_smooth is not None:
        plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
                           max_lag=max_lag, kernel_cells=kde_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation_with_kde, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                                save_dir_img=os.path.join(save_dir_img, 'autocorr.png'))
    else:
        plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells, bin_size=bin_width,
                           max_lag=max_lag)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                                save_dir_img=os.path.join(save_dir_img, 'autocorr.png'))

    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
    #                         xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
    #                         wspace=0.18,
    #                         save_dir_img=os.path.join(save_dir_img2, 'auto_corr_' + str(max_lag) + '.png'))

    # table of peak autocorrelations
    burst_row = ['B' if l else 'N-B' for l in burst_label]
    df = pd.DataFrame(data=[peak_autocorr, burst_row], index=cell_ids, columns=['peak autocorr', 'burst behavior'])
    df.to_csv(os.path.join(save_dir_img, 'peak_autocorr.csv'), index=False)

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
    pl.savefig(os.path.join(save_dir_img, 'peak_autocorr_' + str(max_lag) + '_' + str(bin_width) + '_' + str(
                sigma_smooth) + '.png'))
    pl.show()
