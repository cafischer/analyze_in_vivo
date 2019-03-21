from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import pandas as pd
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from analyze_in_vivo.analyze_domnisoru.autocorr import *
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/autocorr'
    ISIs_cells = load_ISIs()

    # parameters
    bin_width = 0.5  # ms
    max_lag = 12
    sigma_smooth = None  # ms  None for no smoothing
    dt_kde = 0.05  # ms (same as dt data as lower bound for precision)
    t_kde = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
    normalization = 'max'  # 'sum

    folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_'+str(sigma_smooth) + '_normalization_' + str(normalization)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    autocorr_cells = np.zeros((len(ISIs_cells), int(2 * max_lag / bin_width + 1)))
    autocorr_kde_cells = np.zeros((len(ISIs_cells), int(2 * max_lag / dt_kde + 1)))
    kde_cells = np.zeros(len(ISIs_cells), dtype=object)
    peak_autocorr = np.zeros(len(ISIs_cells))
    for cell_idx in range(len(ISIs_cells)):
        print cell_idx
        ISIs = ISIs_cells[cell_idx]

        # get autocorrelation
        autocorr_cells[cell_idx, :], t_autocorr, bins = get_autocorrelation_by_ISIs(ISIs, max_lag=max_lag,
                                                                                    bin_width=bin_width,
                                                                                    normalization=normalization)

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
        # pl.bar(bins[:-1], autocorr_cells[cell_idx, :], bin_width, color='k', align='center', alpha=0.5)
        # pl.plot(t_kde, autocorr_kde_cells[cell_idx, :], color='k')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # pl.show()

    # save peak autocorrelation, autocorrelation
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_kde_cells)
    else:
        np.save(os.path.join(save_dir_img, 'autocorr.npy'), autocorr_cells)
    np.save(os.path.join(save_dir_img, 'peak_autocorr.npy'), peak_autocorr)

    # table of peak autocorrelations
    df = pd.DataFrame(data=np.array([peak_autocorr]).T, index=np.arange(len(ISIs_cells)), columns=['peak autocorr'])
    df.to_csv(os.path.join(save_dir_img, 'peak_autocorr.csv'))


    # plot
    cell_ids = [str(i) for i in range(len(ISIs_cells))]
    cell_type_dict = {str(i): 'not known' for i in cell_ids}
    for n in range(int(np.ceil(len(ISIs_cells) / 27.))):
        end = (n + 1) * 27
        if end >= len(ISIs_cells):
            end = len(ISIs_cells)

        if sigma_smooth is not None:
            plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells[n * 27:end], bin_size=bin_width,
                               max_lag=max_lag, kernel_cells=kde_cells[n * 27:end])
            plot_for_all_grid_cells(cell_ids[n * 27:end], cell_type_dict, plot_autocorrelation_with_kde, plot_kwargs,
                                    xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', legend=False,
                                    save_dir_img=os.path.join(save_dir_img, 'autocorr_'+str(n)+'.png'))
        else:
            plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=autocorr_cells[n * 27:end], bin_size=bin_width,
                               max_lag=max_lag)
            plot_for_all_grid_cells(cell_ids[n * 27:end], cell_type_dict, plot_autocorrelation, plot_kwargs,
                                    xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', legend=False,
                                    save_dir_img=os.path.join(save_dir_img, 'autocorr_'+str(n)+'.png'))
    pl.show()