from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_cumulative_ISI_hist, plot_ISI_hist, plot_cumulative_ISI_hist, \
    plot_cumulative_ISI_hist_all_cells
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.isi import get_ISI_hist_peak_and_width
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
import pandas as pd
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
    ISIs_cells = load_ISIs(save_dir='/home/cf/Phd/programming/data/Caro/grid_cells_withfields_vt_0.pkl')
    max_ISI = 200  # None if you want to take all ISIs
    burst_ISI = 8  # ms
    bin_width = 1  # ms
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    sigma_smooth = 1  # ms  None for no smoothing
    dt_kde = 0.05
    t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)

    folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISI_hist_cells = np.zeros((len(ISIs_cells), len(bins) - 1))
    ISI_kde_cells = np.zeros((len(ISIs_cells), int(max_ISI / dt_kde) + 1))
    cum_ISI_hist_y = [0] * len(ISIs_cells)
    cum_ISI_hist_x = [0] * len(ISIs_cells)
    peak_ISI_hist = np.zeros(len(ISIs_cells), dtype=object)
    width_ISI_hist = np.zeros(len(ISIs_cells))
    kde_cells = np.zeros(len(ISIs_cells), dtype=object)

    for cell_idx in range(len(ISIs_cells)):
        print cell_idx
        ISIs = ISIs_cells[cell_idx]
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]

        # compute KDE
        if sigma_smooth is not None:
            kde_cells[cell_idx] = perform_kde(ISIs, sigma_smooth)
            ISI_kde_cells[cell_idx] = evaluate_kde(t_kde, kde_cells[cell_idx])

        # ISI histograms
        ISI_hist_cells[cell_idx, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs)

        if sigma_smooth is None:
            peak_ISI_hist[cell_idx] = (bins[:-1][np.argmax(ISI_hist_cells[cell_idx, :])],
                                       bins[1:][np.argmax(ISI_hist_cells[cell_idx, :])])
        else:
            peak_ISI_hist[cell_idx], width_ISI_hist[cell_idx] = get_ISI_hist_peak_and_width(ISI_kde_cells[cell_idx], t_kde)

        # plot
        # plot_cumulative_ISI_hist(cum_ISI_hist_x[cell_idx], cum_ISI_hist_y[cell_idx], xlim=(0, max_ISI))
        # print peak_ISI_hist[cell_idx]
        # plot_ISI_hist(ISI_hist_cells[cell_idx, :], bins)
        # pl.show()
        # pl.close('all')

    # # plot all cumulative ISI histograms in one
    # ISIs_all = np.array([item for sublist in ISIs_cells for item in sublist])
    # cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all)
    # plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
    #                                    range(len(ISIs_cells)), max_ISI, False, os.path.join(save_dir_img))
    # pl.show()

    # save
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)
        np.save(os.path.join(save_dir_img, 'width_at_half_ISI_peak.npy'), width_ISI_hist)
        np.save(os.path.join(save_dir_img, 'ISI_hist.npy'), ISI_kde_cells)
    else:
        np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)
        np.save(os.path.join(save_dir_img, 'ISI_hist.npy'), ISI_hist_cells)

    # table of ISI
    df = pd.DataFrame(data=np.vstack((peak_ISI_hist, width_ISI_hist)).T,
                      columns=['ISI peak', 'ISI width'], index=range(len(ISIs_cells)))
    df.index.name = 'Cell ids'
    df.to_csv(os.path.join(save_dir_img, 'ISI_distribution.csv'))