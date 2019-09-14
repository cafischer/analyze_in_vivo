import os
import numpy as np
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import load_cell_ids
from analyze_in_vivo.analyze_domnisoru import perform_kde
from scipy.stats import chisquare

if __name__ == '__main__':
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISIs = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'

    # parameter
    max_ISI = 200
    sigma_smooth = 5
    n_points = 10000
    bin_width = 10
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    folder = 'max_ISI_'+str(max_ISI)+'_bin_width_1'

    # load
    cell_ids = load_cell_ids(save_dir, 'grid_cells')
    ISIs_cells = np.load(os.path.join(save_dir_ISIs, folder, 'ISIs.npy'))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # sample from 1D distribution
        kde = perform_kde(ISIs_cells[cell_idx], sigma_smooth)
        x_1d = kde.resample(n_points)
        y_1d = kde.resample(n_points)

        # sample from 2D distribution
        kde = perform_kde(np.vstack([ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:]]), sigma_smooth)
        x_2d, y_2d = kde.resample(n_points)

        # 2D histogram
        h_1d, _, _ = np.histogram2d(x_1d[0], y_1d[0], bins)
        h_2d, _, _ = np.histogram2d(x_2d, y_2d, bins)

        # statistics
        observed = h_2d.flatten()
        expected = h_1d.flatten()
        idxs = np.where(expected >= 5)[0]
        expected = expected[idxs]
        observed = observed[idxs]

        chisquared, p_val = chisquare(observed, expected, 0)
        print p_val

        # plot
        pl.figure()
        pl.scatter(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], c='k', marker='x', label='real')
        pl.scatter(x_2d, y_2d, c='b', marker='o', alpha=0.5, label='2D')
        pl.scatter(x_1d, y_1d, c='r', marker='o', alpha=0.5, label='1D')
        pl.legend(loc='upper right')

        pl.figure()
        pl.title('Hist 2D - Hist 1D')
        X, Y = np.meshgrid(bins, bins)
        pl.pcolor(X, Y, h_2d-h_1d)
        pl.colorbar()
        pl.show()