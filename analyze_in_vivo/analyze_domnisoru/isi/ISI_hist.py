from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells, plot_cumulative_ISI_hist_all_cells_with_bursty
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax, plot_ISI_hist_on_ax_with_kde, get_ISI_hist_peak_and_width
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
import scipy.stats as st
import pandas as pd
pl.style.use('paper')


if __name__ == '__main__':
    # Note: not all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    #save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    #save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    param_list = ['Vm_ljpc', 'spiketimes']
    max_ISI = 200  # None if you want to take all ISIs
    burst_ISI = 8  # ms
    bin_width = 1  # ms
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    sigma_smooth = 1  # ms  None for no smoothing
    dt_kde = 0.05  # ms
    t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)

    folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_cells = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist_cells = np.zeros((len(cell_ids), len(bins) - 1))
    ISI_kde_cells = np.zeros((len(cell_ids), int(max_ISI / dt_kde) + 1))
    cum_ISI_hist_y = [0] * len(cell_ids)
    cum_ISI_hist_x = [0] * len(cell_ids)
    fraction_ISIs_filtered = np.zeros(len(cell_ids))
    fraction_burst = np.zeros(len(cell_ids))
    peak_ISI_hist = np.zeros(len(cell_ids), dtype=object)
    shortest_ISI = np.zeros(len(cell_ids))
    width_ISI_hist = np.zeros(len(cell_ids))
    CV_ISIs = np.zeros(len(cell_ids))
    kde_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # ISIs
        ISIs = get_ISIs(AP_max_idxs, t)
        if max_ISI is not None:
            fraction_ISIs_filtered[cell_idx] = np.sum(ISIs <= max_ISI) / float(len(ISIs))
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[cell_idx] = len(ISIs)
        ISIs_cells[cell_idx] = ISIs
        fraction_burst[cell_idx] = np.sum(ISIs < burst_ISI) / float(len(ISIs))

        # compute KDE
        if sigma_smooth is not None:
            kde = perform_kde(ISIs, sigma_smooth)
            kde_cells[cell_idx] = kde
            ISI_kde_cells[cell_idx] = evaluate_kde(t_kde, kde)

        # ISI histograms
        ISI_hist_cells[cell_idx, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs)

        if sigma_smooth is None:
            peak_ISI_hist[cell_idx] = (bins[:-1][np.argmax(ISI_hist_cells[cell_idx, :])],
                                       bins[1:][np.argmax(ISI_hist_cells[cell_idx, :])])
        else:
            peak_ISI_hist[cell_idx], width_ISI_hist[cell_idx] = get_ISI_hist_peak_and_width(ISI_kde_cells[cell_idx], t_kde)
        shortest_ISI[cell_idx] = np.mean(np.sort(ISIs)[:int(round(len(ISIs)*0.1))])
        print 'n short: ', int(round(len(ISIs)*0.1))
        print shortest_ISI[cell_idx]

        CV_ISIs[cell_idx] = np.std(ISIs) / np.mean(ISIs)

        # save and plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        #
        # plot_cumulative_ISI_hist(cum_ISI_hist_x[i], cum_ISI_hist_y[i], xlim=(0, 200), title=cell_id,
        #                          save_dir=save_dir_cell)
        # print peak_ISI_hist[cell_idx]
        # plot_ISI_hist(ISI_hist[cell_idx, :], bins, title=cell_id, save_dir=save_dir_cell)
        # pl.show()
        # pl.close('all')

    # save
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'fraction_burst.npy'), fraction_burst)
        np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)
        np.save(os.path.join(save_dir_img, 'width_at_half_ISI_peak.npy'), width_ISI_hist)
        np.save(os.path.join(save_dir_img, 'ISI_hist.npy'), ISI_kde_cells)
    else:
        np.save(os.path.join(save_dir_img, 'fraction_burst.npy'), fraction_burst)
        np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)
        np.save(os.path.join(save_dir_img, 'ISI_hist.npy'), ISI_hist_cells)
        np.save(os.path.join(save_dir_img, 'cum_ISI_hist_y.npy'), cum_ISI_hist_y)
        np.save(os.path.join(save_dir_img, 'cum_ISI_hist_x.npy'), cum_ISI_hist_x)
        np.save(os.path.join(save_dir_img, 'ISIs.npy'), ISIs_cells)
    np.save(os.path.join(save_dir_img, 'shortest_ISI.npy'), shortest_ISI)
    np.save(os.path.join(save_dir_img, 'CV_ISIs.npy'), CV_ISIs)

    # table of ISI
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in cell_ids])
    burst_row = ['B' if l else 'N-B' for l in burst_label]
    df = pd.DataFrame(data=np.vstack((shortest_ISI, peak_ISI_hist, width_ISI_hist, burst_row)).T,
                      columns=['mean(shortest ISIs)', 'ISI peak', 'ISI width', 'burst behavior'], index=cell_ids)
    df.index.name = 'Cell ids'
    df.to_csv(os.path.join(save_dir_img, 'ISI_distribution.csv'))

    # plot all cumulative ISI histograms in one
    # ISIs_all = np.array([item for sublist in ISIs_per_cell for item in sublist])
    # cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all)
    # plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
    #                                                cell_ids, max_ISI, os.path.join(save_dir_img))


    # cumulative ISI histogram for bursty and non-bursty group
    #ISIs_all_bursty =  np.array([item for sublist in np.array(ISIs_per_cell)[burst_label] for item in sublist])
    #ISIs_all_nonbursty = np.array([item for sublist in np.array(ISIs_per_cell)[~burst_label] for item in sublist])
    #cum_ISI_hist_y_avg_bursty, cum_ISI_hist_x_avg_bursty = get_cumulative_ISI_hist(ISIs_all_bursty)
    #cum_ISI_hist_y_avg_nonbursty, cum_ISI_hist_x_avg_nonbursty = get_cumulative_ISI_hist(ISIs_all_nonbursty)
    #plot_cumulative_ISI_hist_all_cells_with_bursty(cum_ISI_hist_y, cum_ISI_hist_x,
    #                                               cum_ISI_hist_y_avg_bursty, cum_ISI_hist_x_avg_bursty,
    #                                               cum_ISI_hist_y_avg_nonbursty, cum_ISI_hist_x_avg_nonbursty,
    #                                               cell_ids, burst_label, max_ISI, os.path.join(save_dir_img2))

    # from scipy.interpolate import UnivariateSpline
    # spline = UnivariateSpline(cum_ISI_hist_x_avg_nonbursty, cum_ISI_hist_y_avg_nonbursty, s=0.005)
    # pl.figure()
    # pl.plot(cum_ISI_hist_x_avg_nonbursty, cum_ISI_hist_y_avg_nonbursty,
    #         drawstyle='steps-post', linewidth=2.0, color='b')
    # x = np.arange(0, 200, 0.01)
    # pl.plot(x, spline(x), 'k')
    # #pl.show()
    #
    # np.diff(spline(x)) / np.diff(x)
    # x[np.where(np.diff(spline(x)) / np.diff(x) > 1 / 200.)[0][0]]
    # print x[np.argmax(np.diff(spline(x)) / np.diff(x))]
    #
    # pl.figure()
    # pl.plot(x[:-1], np.diff(spline(x)) / np.diff(x))
    # #pl.show()
    #
    # spline = UnivariateSpline(cum_ISI_hist_x_avg_bursty, cum_ISI_hist_y_avg_bursty, s=0.005)
    # pl.figure()
    # pl.plot(cum_ISI_hist_x_avg_bursty, cum_ISI_hist_y_avg_bursty,
    #         drawstyle='steps-post', linewidth=2.0, color='r')
    # x = np.arange(0, 200, 0.01)
    # pl.plot(x, spline(x), 'k')
    # #pl.show()
    #
    # np.diff(spline(x)) / np.diff(x)
    # x[np.where(np.diff(spline(x)) / np.diff(x) > 1 / 200.)[0][0]]
    # print x[np.argmax(np.diff(spline(x)) / np.diff(x))]
    #
    # pl.figure()
    # pl.plot(x[:-1], np.diff(spline(x)) / np.diff(x))
    # pl.show()

    # # fraction burst between bursty and non-bursty
    # from scipy.stats import ttest_ind
    # _, p_val = ttest_ind(fraction_burst[burst_label], fraction_burst[~burst_label])

    # # Kolomogorov-Smirnov test between all bursty and non-bursty ISIs
    # D, p_val = ks_2samp(ISIs_all_bursty, ISIs_all_nonbursty)
    # print 'K-S p-value: ', p_val  # p-val small = reject that they come from same distribution
    # # Silhoutte score
    # dist_mat = np.zeros((len(cell_ids), len(cell_ids)))
    # for i1 in range(len(cell_ids)):
    #     for i2 in range(len(cell_ids)):
    #         dist_mat[i1, i2], _ = ks_2samp(ISIs_per_cell[i1], ISIs_per_cell[i2])  # use biggest absolute difference as distance measure
    # silhouette = silhouette_score(dist_mat, burst_label, metric="precomputed")
    # print 'silhouette: ', silhouette

    # # for each pair of cells two sample Kolmogorov Smironov test (Note: ISIs are cut at 200 ms (=max(bins)))
    # p_val_dict = {}
    # for i1, i2 in combinations(range(len(cell_ids)), 2):
    #     D, p_val = ks_2samp(ISIs_per_cell[i1], ISIs_per_cell[i2])
    #     p_val_dict[(i1, i2)] = p_val
    #     print 'p-value for cell '+str(cell_ids[i1]) \
    #           + ' and cell '+str(cell_ids[i2]) + ': %.3f' % p_val
    #
    # plot_cumulative_comparison_all_cells(cum_ISI_hist_x, cum_ISI_hist_y, cell_ids, p_val_dict,
    #                                      os.path.join(save_dir_img, 'comparison_cum_ISI.png'))

    # plot all ISI hists
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    if sigma_smooth is not None:
        plot_kwargs = dict(ISI_hist=ISI_hist_cells, max_ISI=max_ISI, bin_width=bin_width,
                           kernel_cells=kde_cells, dt_kernel=dt_kde)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax_with_kde, plot_kwargs,
                                wspace=0.18, xlabel='ISI (ms)', ylabel='Rel. frequency',
                                save_dir_img=os.path.join(save_dir_img, 'ISI_hist.png'))
    else:
        plot_kwargs = dict(ISI_hist=ISI_hist_cells, cum_ISI_hist_x=cum_ISI_hist_x, cum_ISI_hist_y=cum_ISI_hist_y,
                           max_ISI=max_ISI, bin_width=bin_width)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
                                wspace=0.18, xlabel='ISI (ms)', ylabel='Rel. frequency',
                                save_dir_img=os.path.join(save_dir_img, 'ISI_hist.png'))
        # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
        #                         xlabel='ISI (ms)', ylabel='Rel. frequency', colors_marker=colors_marker,
        #                         wspace=0.18, save_dir_img=os.path.join(save_dir_img2, 'ISI_hist.png'))

    if sigma_smooth is not None:
        def plot_bar(ax, cell_idx, value_cells):
            ax.bar(0.5, value_cells[cell_idx],
                   0.4, color='0.5')
            ax.set_xlim(0, 1)
            ax.set_xticks([])

        plot_kwargs = dict(value_cells=width_ISI_hist)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_bar, plot_kwargs,
                                xlabel='', ylabel='Width at half ISI peak', colors_marker=colors_marker,
                                save_dir_img=os.path.join(save_dir_img, 'width_half_ISI_peak.png'))


    # def plot_bar(ax, cell_idx, value_cells):
    #     ax.bar(0.5, value_cells[cell_idx],
    #            0.4, color='0.5')
    #     ax.set_xlim(0, 1)
    #     ax.set_xticks([])
    #
    # peak_ISI_hist_means = np.array([(a + b) / 2. for (a, b) in peak_ISI_hist])
    # plot_kwargs = dict(value_cells=peak_ISI_hist_means)
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_bar, plot_kwargs,
    #                         xlabel='', ylabel='ISI peak', colors_marker=colors_marker,
    #                         save_dir_img=None)

    # fig, ax = pl.subplots()
    # plot_with_markers(ax, peak_ISI_hist, width_at_half_max, cell_ids, cell_type_dict, theta_cells=theta_cells,
    #                   DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional)
    # ax.set_ylabel('Width at half max (ms)')
    # ax.set_xlabel('Argument of the peak (ms)')
    # pl.show()