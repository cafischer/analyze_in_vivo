from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_cumulative_ISI_hist, plot_ISI_hist, plot_cumulative_ISI_hist, \
    plot_cumulative_ISI_hist_all_cells
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.isi import get_ISI_hist_peak_and_width, plot_ISI_hist_on_ax
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
#pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
    #save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
    ISIs_cells = load_ISIs()
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
    fraction_burst = np.zeros(len(ISIs_cells))
    shortest_ISI = np.zeros(len(ISIs_cells))
    CV_ISIs = np.zeros(len(ISIs_cells))

    for cell_idx in range(len(ISIs_cells)):
        print cell_idx
        ISIs = ISIs_cells[cell_idx]
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]
        fraction_burst[cell_idx] = np.sum(ISIs < burst_ISI) / float(len(ISIs))

        # compute KDE
        if sigma_smooth is not None:
            kde_cells[cell_idx] = perform_kde(ISIs, sigma_smooth)
            ISI_kde_cells[cell_idx] = evaluate_kde(t_kde, kde_cells[cell_idx])

        # ISI histograms
        ISI_hist_cells[cell_idx, :] = get_ISI_hist(ISIs, bins, norm='sum')
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs, max_ISI)

        if sigma_smooth is None:
            peak_ISI_hist[cell_idx] = (bins[:-1][np.argmax(ISI_hist_cells[cell_idx, :])],
                                       bins[1:][np.argmax(ISI_hist_cells[cell_idx, :])])
        else:
            peak_ISI_hist[cell_idx], width_ISI_hist[cell_idx] = get_ISI_hist_peak_and_width(ISI_kde_cells[cell_idx], t_kde)

        shortest_ISI[cell_idx] = np.mean(np.sort(ISIs)[:int(round(len(ISIs) * 0.1))])
        print 'n short: ', int(round(len(ISIs) * 0.1))
        print shortest_ISI[cell_idx]

        CV_ISIs[cell_idx] = np.std(ISIs) / np.mean(ISIs)

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
    np.save(os.path.join(save_dir_img, 'fraction_burst.npy'), fraction_burst)
    np.save(os.path.join(save_dir_img, 'shortest_ISI.npy'), shortest_ISI)
    np.save(os.path.join(save_dir_img, 'CV_ISIs.npy'), CV_ISIs)

    # table of ISI
    df = pd.DataFrame(data=np.vstack((peak_ISI_hist, width_ISI_hist)).T,
                      columns=['ISI peak', 'ISI width'], index=range(len(ISIs_cells)))
    df.index.name = 'Cell ids'
    df.to_csv(os.path.join(save_dir_img, 'ISI_distribution.csv'))

    # plot
    cell_ids = [str(i) for i in range(len(ISIs_cells))]
    cell_type_dict = {str(i): 'not known' for i in cell_ids}
    for n in range(int(np.ceil(len(ISIs_cells) / 27.))):
        end = (n + 1) * 27
        if end >= len(ISIs_cells):
            end = len(ISIs_cells)
        plot_kwargs = dict(ISI_hist=ISI_hist_cells[n * 27:end],
                           cum_ISI_hist_x=cum_ISI_hist_x[n * 27:end], cum_ISI_hist_y=cum_ISI_hist_y[n * 27:end],
                           max_ISI=max_ISI, bin_width=bin_width)
        plot_for_all_grid_cells(cell_ids[n * 27:end], cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
                                xlabel='ISI (ms)', ylabel='Rel. frequency', legend=False, wspace=0.18,
                                save_dir_img=os.path.join(save_dir_img, 'ISI_hist_' + str(n) + '.png'))
    pl.show()