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
#pl.style.use('paper_subplots')


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

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

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
    pl.show()