from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from scipy.stats import ks_2samp
from itertools import combinations
pl.style.use('paper')



if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies

    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'pyramidal_layer2'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    filter_long_ISIs = False
    max_ISI = 200
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(max_ISI))

    # parameter
    bins = np.arange(0, max_ISI+2, 2.0)

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist_y = [0] * len(cell_ids)
    cum_ISI_hist_x = [0] * len(cell_ids)

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt, interval=2, v_diff_onset_max=5)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[i] = len(ISIs)
        ISIs_per_cell[i] = ISIs

        # ISI histograms
        ISI_hist[i, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[i], cum_ISI_hist_x[i] = get_cumulative_ISI_hist(ISIs)

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        plot_cumulative_ISI_hist(cum_ISI_hist_x[i], cum_ISI_hist_y[i], xlim=(0, 200), title=cell_id,
                                 save_dir=save_dir_cell)
        plot_ISI_hist(ISI_hist[i, :], bins, title=cell_id, save_dir=save_dir_cell)
        #pl.show()
        pl.close('all')

    # plot all cumulative ISI histograms in one
    ISIs_all = np.array([item for sublist in ISIs_per_cell for item in sublist])
    cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all)
    plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
                                       cell_ids, max_ISI, os.path.join(save_dir_img, cell_type))

    # for each pair of cells two sample Kolmogorov Smironov test (Note: ISIs are cut at 200 ms (=max(bins)))
    p_val_dict = {}
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        D, p_val = ks_2samp(ISIs_per_cell[i1], ISIs_per_cell[i2])
        p_val_dict[(i1, i2)] = p_val
        print 'p-value for cell '+str(cell_ids[i1]) \
              + ' and cell '+str(cell_ids[i2]) + ': %.3f' % p_val

    plot_cumulative_comparison_all_cells(cum_ISI_hist_x, cum_ISI_hist_y, cell_ids, p_val_dict,
                                         os.path.join(save_dir_img, cell_type, 'comparison_cum_ISI.png'))