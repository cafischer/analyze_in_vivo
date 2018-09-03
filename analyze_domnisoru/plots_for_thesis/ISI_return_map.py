from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_ids_grid = np.array(load_cell_ids(save_dir, 'grid_cells'))
    filter_long_ISIs = True
    max_ISI = 200  # ms
    bin_width = 1.0  # ms
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    good_AP_DAP_cell_ids = get_cell_ids_DAP_cells()
    good_AP_no_DAP_cell_ids = ['s74_0006', 's82_0002']
    bad_AP_no_DAP_cell_ids = ['s73_0004', 's95_0006', 's85_0007']
    cell_ids = good_AP_DAP_cell_ids + good_AP_no_DAP_cell_ids + bad_AP_no_DAP_cell_ids
    cell_idxs = [np.where(cell_id == cell_ids_grid)[0][0] for cell_id in cell_ids]
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    ISIs_cells = np.zeros(len(cell_ids_grid), dtype=object)
    ISI_hist = np.zeros((len(cell_ids_grid), len(bins)-1))
    cum_ISI_hist_y = np.zeros(len(cell_ids_grid), dtype=object)
    cum_ISI_hist_x = np.zeros(len(cell_ids_grid), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids_grid):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        AP_max_idxs = data['spiketimes']

        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_cells[cell_idx] = ISIs

    # plot
    fig = pl.figure(figsize=(9.5, 5.5))
    n_rows, n_columns = 2, 5
    outer = gridspec.GridSpec(n_rows, n_columns, hspace=0.3)

    # ISI return map
    axes = [outer[0, i] for i in range(5)] + [outer[1, i] for i in range(5)]
    for i, (cell_idx, cell_id) in enumerate(zip(cell_idxs, cell_ids)):

        ax1 = pl.subplot(axes[i])
        fig.add_subplot(ax1)
        ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
        ax1.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
                 marker='o', linestyle='', markersize=1, alpha=0.5)
        ax1.set_xticks([0, 100, 200])
        ax1.set_yticks([0, 100, 200])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal', adjustable='box-forced')

        if i == 0 or i == 5:
            ax1.set_ylabel('ISI[n+1] (ms)')
            ax1.set_yticklabels([0, 100, 200])
        if i >= 5:
            ax1.set_xlabel('ISI[n] (ms)')
            ax1.set_xticklabels([0, 100, 200])

    # title
    ax1.annotate('DAP', xy=(0.53, 0.96), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')
    ax1.annotate('No DAP', xy=(0.53, 0.49), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')

    pl.tight_layout()
    pl.subplots_adjust(top=0.94, bottom=0.09, left=0.07, right=0.96)
    pl.savefig(os.path.join(save_dir_img, 'ISI_return_map.png'))

    pl.show()