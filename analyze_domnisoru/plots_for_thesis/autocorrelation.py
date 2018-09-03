from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    cell_ids_grid = np.array(load_cell_ids(save_dir, 'grid_cells'))
    max_lag = 50
    bin_size = 1.0  # ms
    auto_corr_cells = np.load(os.path.join(save_dir_auto_corr, 'grid_cells', 'auto_corr_'+str(max_lag)+'.npy'))
    max_lag_idx = to_idx(max_lag, bin_size)
    t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_size

    good_AP_DAP_cell_ids = get_cell_ids_DAP_cells()
    good_AP_no_DAP_cell_ids = ['s74_0006', 's82_0002']
    bad_AP_no_DAP_cell_ids = ['s73_0004', 's95_0006', 's85_0007']
    cell_ids = good_AP_DAP_cell_ids + good_AP_no_DAP_cell_ids + bad_AP_no_DAP_cell_ids
    cell_idxs = [np.where(cell_id == cell_ids_grid)[0][0] for cell_id in cell_ids]
    cell_type_dict = get_celltype_dict(save_dir)

    # parameters
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # plot
    fig = pl.figure(figsize=(9.5, 5.5))
    n_rows, n_columns = 2, 5
    outer = gridspec.GridSpec(n_rows, n_columns, hspace=0.3)

    # autocorrelation
    axes = [outer[0, i] for i in range(5)] + [outer[1, i] for i in range(5)]
    for i, (cell_idx, cell_id) in enumerate(zip(cell_idxs, cell_ids)):

        ax1 = pl.subplot(axes[i])
        fig.add_subplot(ax1)
        ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
        ax1.bar(t_auto_corr, auto_corr_cells[cell_idx] / np.max(auto_corr_cells[cell_idx]), bin_size, color='0.5',
                align='center')
        ax1.set_xticks([-50, 0, 50])
        ax1.set_yticks([0, 1])

        if i == 0 or i == 5:
            ax1.set_ylabel('Spike-time \nautocorrelation (norm.)')
        if i >= 5:
            ax1.set_xlabel('Lag (ms)')

    # title
    ax1.annotate('DAP', xy=(0.53, 0.96), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')
    ax1.annotate('No DAP', xy=(0.53, 0.49), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')

    pl.tight_layout()
    pl.subplots_adjust(top=0.92, bottom=0.09, left=0.08, right=0.98)
    pl.savefig(os.path.join(save_dir_img, 'autocorrelation.png'))

    pl.show()