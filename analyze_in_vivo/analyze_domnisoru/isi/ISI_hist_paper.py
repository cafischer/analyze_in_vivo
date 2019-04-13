from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax, plot_ISI_return_map
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    #save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'

    cell_ids_grid = np.array(load_cell_ids(save_dir, 'grid_cells'))
    max_ISI = 200  # ms
    bin_width = 1  # ms
    sigma_smooth = None  # ms
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    cell_ids_plot = ['s76_0002', 's109_0002', 's84_0002']
    cell_idxs = [np.where(cell_id == cell_ids_grid)[0][0] for cell_id in cell_ids_plot]
    cell_type_dict = get_celltype_dict(save_dir)

    #if not os.path.exists(save_dir_img):
    #    os.makedirs(save_dir_img)

    folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
    ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, folder, 'ISI_hist.npy'))
    cum_ISI_hist_y = np.load(os.path.join(save_dir_ISI_hist, folder, 'cum_ISI_hist_y.npy'))
    cum_ISI_hist_x = np.load(os.path.join(save_dir_ISI_hist, folder, 'cum_ISI_hist_x.npy'))
    ISIs_cells = np.load(os.path.join(save_dir_ISI_hist, folder, 'ISIs.npy'))

    # plot
    fig = pl.figure(figsize=(9.5, 5.5))
    n_rows, n_columns = 2, 3
    outer = gridspec.GridSpec(n_rows, n_columns, hspace=0.3)

    for i, (cell_idx, cell_id) in enumerate(zip(cell_idxs, cell_ids_plot)):
        # ISI hist.
        ax = pl.subplot(outer[0, i])
        fig.add_subplot(ax)
        #ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
        plot_ISI_hist_on_ax(ax, cell_idx, ISI_hist_cells, cum_ISI_hist_x, cum_ISI_hist_y, max_ISI, bin_width)
        ax.set_ylim(0, 0.3)
        if i == 0:
            ax.set_ylabel('Rel. frequency')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('ISI (ms)')

        if i == 0:
            ax.text(-0.35, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')

        # ISI return map
        ax = pl.subplot(outer[1, i])
        fig.add_subplot(ax)
        plot_ISI_return_map(ax, cell_idx, ISIs_cells, max_ISI)
        if i == 0:
            ax.set_ylabel('ISI[n+1] (ms)')
        if i != 0:
            ax.set_yticklabels([])
        ax.set_xlabel('ISI[n] (ms)')

        if i == 0:
            ax.text(-0.5, 1.0, 'B', transform=ax.transAxes, size=18, weight='bold')

    #pl.subplots_adjust(top=0.92, bottom=0.09, left=0.07, right=0.96)
    pl.savefig(os.path.join(save_dir_img, 'ISI_hist.png'))
    pl.show()