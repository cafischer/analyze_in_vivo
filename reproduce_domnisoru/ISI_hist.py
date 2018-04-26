from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_spike_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from scipy.stats import ks_2samp
from itertools import combinations, product
pl.style.use('paper')


def plot_cum_ISI_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg, cell_ids, save_dir):
    max_x = 200
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    cum_ISI_hist_x_avg_with_end = np.insert(cum_ISI_hist_x_avg, len(cum_ISI_hist_x_avg), max_x)
    cum_ISI_hist_y_avg_with_end = np.insert(cum_ISI_hist_y_avg, len(cum_ISI_hist_y_avg), 1.0)
    pl.figure()
    pl.plot(cum_ISI_hist_x_avg_with_end, cum_ISI_hist_y_avg_with_end, label='all',
            drawstyle='steps-post', linewidth=2.0, color='0.4')
    for i, cell_id in enumerate(cell_ids):
        cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[i], len(cum_ISI_hist_x[i]), max_x)
        cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[i], len(cum_ISI_hist_y[i]), 1.0)
        pl.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, label=cell_id, color=colors[i],
                drawstyle='steps-post')
    pl.xlabel('ISI (ms)')
    pl.ylabel('CDF')
    pl.xlim(0, max_x)
    pl.legend(fontsize=10, loc='lower right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'cum_ISI_hist.png'))


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies

    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_ids = load_cell_ids(save_dir, 'stellate_layer2')
    param_list = ['Vm_ljpc']
    AP_thresholds = {'s117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50,
                     's101_0009': -45}
    filter_long_ISIs = False
    filter_long_ISIs_max = 200
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(filter_long_ISIs_max))

    # parameter
    bins = np.arange(0, 200+2, 2.0)

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
        AP_max_idxs = get_spike_idxs(v, AP_thresholds[cell_id], dt, interval=2, v_diff_onset_max=5)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= filter_long_ISIs_max]
        n_ISIs[i] = len(ISIs)
        ISIs_per_cell[i] = ISIs

        # ISI histograms
        ISI_hist[i, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[i], cum_ISI_hist_x[i] = get_cumulative_ISI_hist(ISIs)

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
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
    plot_cum_ISI_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg, cell_ids, save_dir_img)

    # for each pair of cells two sample Kolmogorov Smironov test (Note: ISIs are cut at 200 ms (=max(bins)))
    save_dir_ks = os.path.join(save_dir_img, 'kolmogorov_smirnov_2cells')
    if not os.path.exists(save_dir_ks):
        os.makedirs(save_dir_ks)

    p_val_dict = {}
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        D, p_val = ks_2samp(ISIs_per_cell[i1], ISIs_per_cell[i2])
        p_val_dict[(i1, i2)] = p_val
        print 'p-value for cell '+str(cell_ids[i1]) \
              + ' and cell '+str(cell_ids[i2]) + ': %.3f' % p_val

    fig, ax = pl.subplots(len(cell_ids)-1, len(cell_ids)-1, figsize=(10, 10))
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    for i1, i2 in product(range(len(cell_ids)-1), repeat=2):
        ax[i2-1, i1].spines['left'].set_visible(False)
        ax[i2-1, i1].spines['bottom'].set_visible(False)
        ax[i2-1, i1].set_xticks([])
        ax[i2-1, i1].set_yticks([])
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        ax[i2-1, i1].set_title('p-value: %.3f' % p_val_dict[(i1, i2)])
        ax[i2-1, i1].plot(cum_ISI_hist_x[i1], cum_ISI_hist_y[i1], color=colors[i1])
        ax[i2-1, i1].plot(cum_ISI_hist_x[i2], cum_ISI_hist_y[i2], color=colors[i2])
        ax[i2-1, i1].set_xlim(0, 200)
        ax[i2-1, i1].set_xticks([0, 100, 200])
        ax[i2-1, i1].set_yticks([0, 1])
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        ax[i2-1, i1].set(xlabel=cell_ids[i1], ylabel=cell_ids[i2])
        ax[i2-1, i1].label_outer()
        ax[i2-1, i1].spines['left'].set_visible(True)
        ax[i2-1, i1].spines['bottom'].set_visible(True)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_ks, 'comparison_cum_ISI.png'))
    pl.show()