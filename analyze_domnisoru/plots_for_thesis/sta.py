from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.sta import plot_sta, plot_v_hist
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from mpl_toolkits.mplot3d import Axes3D
pl.style.use('paper_subplots')


# bad_AP_no_DAP_all = ['s43_0003' 's67_0000' 's73_0004' 's76_0002' 's81_0004' 's84_0002'
#  's85_0007' 's90_0006' 's95_0006' 's96_0009' 's100_0006' 's101_0009'
#  's115_0018' 's115_0024' 's115_0030' 's117_0002' 's118_0002' 's120_0002'
#  's120_0023']

if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/not_detrended/all/grid_cells'
    save_dir_sta_good_APs = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/not_detrended/all/grid_cells'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    good_AP_DAP_cell_ids = get_cell_ids_DAP_cells()
    good_AP_no_DAP_cell_ids = ['s74_0006', 's82_0002']

    bad_AP_no_DAP_cell_ids = ['s73_0004', 's95_0006', 's85_0007']
    cell_ids = good_AP_DAP_cell_ids + good_AP_no_DAP_cell_ids + bad_AP_no_DAP_cell_ids
    cell_type_dict = get_celltype_dict(save_dir)

    # parameters
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    v_hist_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        save_dir_cell = os.path.join(save_dir_sta, cell_id)

        sta_mean_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
        sta_std_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
        v_hist_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'v_hist.npy'))
        t_AP = np.load(os.path.join(save_dir_cell, 't_AP.npy'))
        bins_v = np.load(os.path.join(save_dir_cell, 'bins_v.npy'))
        # sta_mean_cells[cell_idx], sta_std_cells[cell_idx], v_hist_cells[cell_idx], t_AP = get_sta_for_cell_id(cell_id,
        #                                                                                                      param_list,
        #                                                                                                      save_dir)

    # plot
    fig = pl.figure(figsize=(9.5, 8.5))
    n_rows, n_columns = 2, 5
    outer = gridspec.GridSpec(n_rows, n_columns, hspace=0.3)

    # sta and time-resolved hist.
    axes = [outer[0, i] for i in range(5)] + [outer[1, i] for i in range(5)]
    for cell_idx in range(len(cell_ids)):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=axes[cell_idx], hspace=0.05)

        ax1 = pl.subplot(inner[0])
        fig.add_subplot(ax1)
        ax1.set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))
        plot_sta(ax1, cell_idx, t_AP, sta_mean_cells, sta_std_cells)
        ax1.set_ylim(-75, 15)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = pl.subplot(inner[1])
        plot_v_hist(ax2, cell_idx, t_AP, bins_v, v_hist_cells)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylim(-75, 15)
        ax2.set_yticks([])

        if cell_idx == 0 or cell_idx == 5:
            ax1.set_ylabel('Mem. pot. (mV)')
            ax2.set_ylabel('Mem. pot. \ndistr. (mV)')
            ax1.set_yticks([-60, -40, -20, 0])
            ax2.set_yticks([-60, -40, -20, 0])

    # title
    ax1.annotate('DAP', xy=(0.53, 0.967), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')
    ax2.annotate('No DAP', xy=(0.53, 0.471), xycoords='figure fraction', fontsize=14,
                 horizontalalignment='center')

    pl.tight_layout()
    pl.subplots_adjust(top=0.94, bottom=0.06, left=0.1, right=0.98)
    pl.savefig(os.path.join(save_dir_img, 'sta.png'))

    pl.show()