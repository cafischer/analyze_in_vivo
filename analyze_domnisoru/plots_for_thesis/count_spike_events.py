from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.n_spikes_in_burst import get_n_spikes_in_burst
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_ids_grid = np.array(load_cell_ids(save_dir, 'grid_cells'))
    filter_long_ISIs = True
    ISI_burst = 8.0  # ms 
    bins = np.arange(1, 10 + 1, 1)

    DAP_cell_ids = get_cell_ids_DAP_cells()
    other_examplses = ['s73_0004', 's95_0006', 's76_0002', 's74_0006', 's85_0007']
    cell_ids = DAP_cell_ids + other_examplses
    cell_idxs = [np.where(cell_id == cell_ids_grid)[0][0] for cell_id in cell_ids]
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    count_spikes = np.zeros((len(cell_ids_grid), len(bins) - 1))

    for cell_idx, cell_id in enumerate(cell_ids_grid):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # find burst indices
        AP_max_idxs = data['spiketimes']
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_spikes_in_bursts = get_n_spikes_in_burst(short_ISI_indicator.astype(int))
        count_spikes[cell_idx, :] = np.histogram(n_spikes_in_bursts, bins)[0]

        starts_burst, ends_burst = get_starts_ends_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))
        count_spikes[cell_idx, 0] = len(AP_max_idxs_single)
        assert bins[0] == 1

    # plot
    fig = pl.figure(figsize=(9.5, 5.5))
    n_rows, n_columns = 2, 5
    outer = gridspec.GridSpec(n_rows, n_columns, hspace=0.3)

    # ISI return map
    axes = [outer[0, i] for i in range(5)] + [outer[1, i] for i in range(5)]
    for i, (cell_idx, cell_id) in enumerate(zip(cell_idxs, cell_ids)):

        ax = pl.subplot(axes[i])
        fig.add_subplot(ax)
        ax.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
        ax.bar(bins[:-1], count_spikes[cell_idx, :] / np.max(count_spikes[cell_idx, :]),
               color='0.5')
        ax.set_xlim(bins[0] - 0.5, bins[-1])
        ax.set_ylim(0, 1)
        ax.set_xticks(bins)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax_twin = ax.twinx()
        # ax_twin.plot(bins[:-1], np.log(count_spikes[cell_idx, :]) / np.max(np.log(count_spikes[cell_idx, :])),
        #         marker='o', linestyle='-', color='k', markersize=5)
        ax_twin.plot(bins[:-1], count_spikes[cell_idx, :] / np.max(count_spikes[cell_idx, :]),
                     marker='o', linestyle='-', color='k', markersize=5)
        ax_twin.set_yscale('log')
        if not(i == 4 or i == 9):
            ax_twin.set_yticklabels([])
        ax.spines['right'].set_visible(True)

        if i == 0 or i == 5:
            ax.set_ylabel('Rel. frequency')
            ax.set_yticklabels(['%.2f' % j for j in np.arange(0, 1.1, 0.2)])
        if i >= 5:
            ax.set_xlabel('# Spikes in event')
            labels = [''] * len(bins)
            labels[::2] = bins[::2]
            ax.set_xticklabels(labels)
        if i == 4 or i == 9:
            ax_twin.set_ylabel('Rel. log. frequency')

    # title
    ax.annotate('DAP', xy=(0.53, 0.96), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')
    ax.annotate('Other examples', xy=(0.53, 0.49), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')

    pl.tight_layout()
    pl.subplots_adjust(top=0.92, bottom=0.09, left=0.08, right=0.92)
    pl.savefig(os.path.join(save_dir_img, 'count_spike_events.png'))

    pl.show()