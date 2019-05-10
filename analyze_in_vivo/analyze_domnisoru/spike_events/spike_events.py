from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from analyze_in_vivo.analyze_domnisoru.spike_events import get_starts_ends_burst, get_idxs_single, get_burst_lengths
from scipy.stats import chisquare
from scipy.optimize import curve_fit
import pandas as pd
pl.style.use('paper')


def plot_n_spikes_in_burst_all_cells(cell_type_dict, bins, count_spikes):
    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    if cell_type == 'grid_cells':
        def plot_fun(ax, cell_idx, bins, count_spikes):
            count_spikes_normed = count_spikes[cell_idx, :] / (np.sum(count_spikes[cell_idx, :]) * (bins[1] - bins[0]))
            ax.bar(bins[:-1], count_spikes_normed, color='0.5')
            ax.set_xlim(bins[0] - 0.5, bins[-1])
            ax.set_xticks(bins)
            labels = [''] * len(bins)
            labels[::4] = bins[::4]
            ax.set_xticklabels(labels)

            # with log scale
            ax_twin = ax.twinx()
            ax_twin.plot(bins[:-1], count_spikes_normed, marker='o', linestyle='-', color='k', markersize=3)
            ax_twin.set_yscale('log')
            ax_twin.set_ylim(10**-4, 10**0)
            if not (cell_idx == 8 or cell_idx == 17 or cell_idx == 25):
                ax_twin.set_yticklabels([])
            else:
                ax_twin.set_ylabel('Rel. log. frequency')
            ax.spines['right'].set_visible(True)

        burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
        colors_marker = np.zeros(len(burst_label), dtype=str)
        colors_marker[burst_label] = 'r'
        colors_marker[~burst_label] = 'b'

        plot_kwargs = dict(bins=bins, count_spikes=count_spikes)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs,
                                xlabel='# Spikes \nin event', ylabel='Rel. frequency',
                                colors_marker=colors_marker, wspace=0.18,
                                save_dir_img=os.path.join(save_dir_img, 'count_spikes_' + str(burst_ISI) + '.png'))

        # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs,
        #                         xlabel='# Spikes \nin event', ylabel='Rel. frequency',
        #                         colors_marker=colors_marker, wspace=0.18,
        #                         save_dir_img=os.path.join(save_dir_img2, 'count_spikes.png'))


def geom_dist(x, rate):
    p = 1 - np.exp(-rate)
    return (1-p)**(x-1) * p


def test_geom_dist_with_chi_square(count_events, x_events):
    observed = count_events
    x = x_events
    # sample data for testing
    # samples = np.random.geometric(1 - np.exp(-2.), 1000)
    # observed = np.histogram(samples, bins)[0]

    # fit geometric distribution to the data
    n_estimated_parameters = 1
    p_opt, _ = curve_fit(geom_dist, x, observed / np.sum(observed))
    rate_est = p_opt[0]
    expected = geom_dist(x, rate_est) * np.sum(observed)

    # add to the first bin with expected count per bin < 4 the counts from the following bins
    if not np.all(expected >= 4):
        idx = np.where(expected < 4)[0][0] - 1
        observed = np.append(observed[:idx], np.sum(observed[idx:]))
        expected = np.append(expected[:idx], np.sum(expected[idx:]))

    # compute statistic
    chisquared, p_val = chisquare(observed, expected, n_estimated_parameters)

    # print 'rate est.: ', rate_est
    # print 'chi-squared: %.2f' % chisquared
    # print 'p-val: %.5f' % p_val
    # if len(observed) > 1:
    #     pl.figure()
    #     pl.bar(np.arange(len(observed_)), observed_, width=1)
    #     pl.plot(np.arange(len(expected_)), expected_, 'or')
    #     pl.show()
    return p_val


def doublet_prevalence_assuming_geometric(count_events):
    p1 = count_events[0] / float(np.sum(count_events))
    p2 = count_events[1] / float(np.sum(count_events))
    p3 = count_events[2] / float(np.sum(count_events))
    return p2**2 - p1 * p3

# not sig. different with alpha 0.05
# s79_0003
# s101_0009
# s118_0002
# s119_0004
# s120_0002

if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    #save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    #save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    bins = np.arange(1, 15 + 1, 1)
    burst_ISI = 8  # ms

    count_spikes = np.zeros((len(cell_ids), len(bins)-1))
    fraction_single = np.zeros(len(cell_ids))

    p_vals = np.zeros(len(cell_ids))
    prevalence_doublets = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # get APs
        AP_max_idxs = data['spiketimes']

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        burst_ISI_indicator = ISIs <= burst_ISI
        starts_burst, ends_burst = get_starts_ends_burst(burst_ISI_indicator)
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = AP_max_idxs[get_idxs_single(burst_ISI_indicator, ends_burst)]
        count_spikes[cell_idx, :] = np.histogram(get_burst_lengths(starts_burst, ends_burst), bins)[0]
        count_spikes[cell_idx, 0] = len(AP_max_idxs_single)
        assert bins[0] == 1
        fraction_single[cell_idx] = count_spikes[cell_idx, 0] / np.sum(count_spikes[cell_idx, :])

        # test for geometric distribution
        p_vals[cell_idx] = test_geom_dist_with_chi_square(count_spikes[cell_idx], bins[:-1])
        print cell_id
        print 'p-val: %.3f' % p_vals[cell_idx]
        prevalence_doublets[cell_idx] = doublet_prevalence_assuming_geometric(count_spikes[cell_idx])
        print '[p(2)]^2-p(1)*p(3): %.2f' % prevalence_doublets[cell_idx]

        # pl.close('all')
        # pl.figure()
        # pl.bar(bins[:-1], count_spikes[cell_idx, :], color='0.5')
        # pl.xlabel('# Spikes')
        # pl.ylabel('Frequency')
        # pl.xticks(bins)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'n_spikes_in_burst.png'))
        # pl.show()

    # fraction single between bursty and non-bursty
    # from scipy.stats import ttest_ind
    # burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    # _, p_val = ttest_ind(fraction_single[burst_label], fraction_single[~burst_label])
    # print 'p_val: ', p_val
    # pl.figure()
    # pl.plot(np.zeros(sum(burst_label)), fraction_single[burst_label], 'or')
    # pl.plot(np.ones(sum(~burst_label)), fraction_single[~burst_label], 'ob')

    # plot all cells
    np.save(os.path.join(save_dir_img, 'fraction_single_' + str(burst_ISI) + '.npy'), fraction_single)

    pl.close('all')
    plot_n_spikes_in_burst_all_cells(cell_type_dict, bins, count_spikes)
    #pl.show()

df = pd.DataFrame(data=np.vstack((p_vals, prevalence_doublets)).T,
                  columns=['p-val (chi-square)', '[p(2)]^2-p(1)*p(3)'], index=cell_ids)
df.index.name = 'Cell ID'
df = df.astype(float).round(3)
df.to_csv(os.path.join(save_dir_img, 'spike_events.csv'))