from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells, plot_cumulative_comparison_all_cells
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from scipy.stats import ks_2samp
from itertools import combinations
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = False
    max_ISI = 200
    burst_ISI = 8  # ms
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(max_ISI))
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # parameter
    bin_width = 1.0
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist_y = [0] * len(cell_ids)
    cum_ISI_hist_x = [0] * len(cell_ids)
    fraction_ISIs_filtered = np.zeros(len(cell_ids))
    len_recording = np.zeros(len(cell_ids))
    firing_rate = np.zeros(len(cell_ids))
    fraction_burst = np.zeros(len(cell_ids))
    peak_ISI_hist = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            fraction_ISIs_filtered[cell_idx] = np.sum(ISIs <= max_ISI) / float(len(ISIs))
            len_recording[cell_idx] = t[-1]
            firing_rate[cell_idx] = len(AP_max_idxs) / (len_recording[cell_idx] / 1000.0)
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[cell_idx] = len(ISIs)
        ISIs_per_cell[cell_idx] = ISIs
        fraction_burst[cell_idx] = np.sum(ISIs < burst_ISI) / float(len(ISIs))

        # ISI histograms
        ISI_hist[cell_idx, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs)
        peak_ISI_hist[cell_idx] = (bins[:-1][np.argmax(ISI_hist[cell_idx, :])],
                                   bins[1:][np.argmax(ISI_hist[cell_idx, :])])

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

    # plot all cumulative ISI histograms in one
    ISIs_all = np.array([item for sublist in ISIs_per_cell for item in sublist])
    cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all)
    plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
                                       cell_ids, max_ISI, os.path.join(save_dir_img))

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

    # save
    np.save(os.path.join(save_dir_img, 'fraction_burst.npy'), fraction_burst)
    np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)

    # plot all ISI hists
    if cell_type == 'grid_cells':
        def plot_ISI_hist(ax, cell_idx, fraction_ISIs_filtered, ISI_hist, cum_ISI_hist_x, cum_ISI_hist_y):
            if filter_long_ISIs:
                ax.annotate('%i%%<200 ms' % int(round(fraction_ISIs_filtered[cell_idx] * 100)),
                                      xy=(0.07, 0.98), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                                      bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))

            ax.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
                             bins[1] - bins[0], color='0.5', align='edge')
            cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[cell_idx], len(cum_ISI_hist_x[cell_idx]), max_ISI)
            cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[cell_idx], len(cum_ISI_hist_y[cell_idx]), 1.0)
            ax_twin = ax.twinx()
            ax_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
            ax_twin.set_xlim(0, max_ISI)
            ax_twin.set_ylim(0, 1)
            if (cell_idx+1) % 9 == 0:
                ax_twin.set_yticks([0, 1])
            else:
                ax_twin.set_yticks([])
            ax.spines['right'].set_visible(True)

        plot_kwargs = dict(fraction_ISIs_filtered=fraction_ISIs_filtered, ISI_hist=ISI_hist,
                           cum_ISI_hist_x=cum_ISI_hist_x, cum_ISI_hist_y=cum_ISI_hist_y)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist, plot_kwargs,
                                xlabel='ISI (ms)', ylabel='Rel. frequency',
                                save_dir_img=os.path.join(save_dir_img, 'ISI_hist'+str(bin_width)+'.png'))


        def plot_fraction_burst(ax, cell_idx, fraction_burst):
            ax.bar(0.5, fraction_burst[cell_idx],
                   0.4, color='0.5')
            ax.set_xlim(0, 1)
            ax.set_xticks([])

        plot_kwargs = dict(fraction_burst=fraction_burst)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fraction_burst, plot_kwargs,
                                xlabel='', ylabel='Fraction burst',
                                save_dir_img=os.path.join(save_dir_img, 'fraction_burst' + str(bin_width) + '.png'))

    pl.figure()
    pl.plot(len_recording / 1000.0, fraction_ISIs_filtered, 'ok')
    pl.xlabel('Dur. recording (s)')
    pl.ylabel('Fraction ISI < 200 ms')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'dur_rec_vs_fraction_ISI.png'))

    pl.figure()
    pl.plot(firing_rate / 1000.0, fraction_ISIs_filtered, 'ok')
    pl.xlabel('Firing rate (Hz)')
    pl.ylabel('Fraction ISI < 200 ms')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'firing_rate_vs_fraction_ISI.png'))
    #pl.show()