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


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/not_detrended/all/grid_cells'
    save_dir_sta_good_APs = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/not_detrended/all/grid_cells'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    cell_type = 'DAP_cells'
    cell_ids = get_cell_ids_DAP_cells()
    cell_type_dict = get_celltype_dict(save_dir)

    # parameters
    use_AP_max_idxs_domnisoru = True
    param_list = ['Vm_ljpc', 'spiketimes']
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
    outer = gridspec.GridSpec(n_rows, n_columns, height_ratios=[0.65, 0.35])

    # sta and time-resolved hist.
    for cell_idx in range(len(cell_ids)):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, cell_idx], hspace=0.05)

        ax1 = pl.subplot(inner[0])
        fig.add_subplot(ax1)
        ax1.set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))
        plot_sta(ax1, cell_idx, t_AP, sta_mean_cells, sta_std_cells)
        ax1.set_ylim(-75, 15)
        ax1.set_xticks([])

        ax2 = pl.subplot(inner[1])
        plot_v_hist(ax2, cell_idx, t_AP, bins_v, v_hist_cells)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylim(-75, 15)

        if cell_idx == 0:
            ax1.set_ylabel('Mem. pot. (mV)')
            ax2.set_ylabel('Mem. pot. \ndistr. (mV)')

    # goodness of recording
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, :3], width_ratios=[0.9, 0.1])
    ax = pl.subplot(inner[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax = pl.subplot(inner[0], projection='3d')
    #ax = pl.subplot(outer[1, :2], projection='3d')
    grid_cells = load_cell_ids(save_dir, 'grid_cells')
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells()
    DAP_deflection = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_deflection.npy'))
    DAP_deflection[np.isnan(DAP_deflection)] = 0
    AP_width = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_width.npy'))
    AP_amp = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_amp.npy'))
    plot_with_markers(ax, AP_width, AP_amp, np.array(grid_cells), cell_type_dict, DAP_deflection, 'k',
                      theta_cells, DAP_cells)
    ax.set_xlabel('AP width (ms)')
    ax.set_ylabel('AP amp. (mV)')
    ax.set_zlabel('DAP deflection (mV)')
    ax.view_init(elev=28, azim=38)
    l = ax.get_legend()
    l.set_bbox_to_anchor((1.0, 0.9))

    # example select good APs
    cell_id = 's73_0004'
    save_dir_cell = os.path.join(save_dir_sta, cell_id)
    sta_mean = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    sta_std = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    v_hist = np.load(os.path.join(save_dir_cell, 'v_hist.npy'))
    t_AP = np.load(os.path.join(save_dir_cell, 't_AP.npy'))
    save_dir_cell = os.path.join(save_dir_sta_good_APs, cell_id)
    sta_mean_good_APs = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    sta_std_good_APs = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))

    ax1 = pl.subplot(outer[1, 3])
    fig.add_subplot(ax1)
    ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    ax1.annotate('all APs', xy=(t_AP[0], 15), textcoords='data',
                horizontalalignment='left', verticalalignment='top', fontsize=9)
    plot_sta(ax1, 0, t_AP, [sta_mean], [sta_std])
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Mem. pot. (mV)')
    ax1.set_ylim(-75, 15)

    ax2 = pl.subplot(outer[1, 4])
    ax2.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    ax2.annotate('selected APs', xy=(t_AP[0], 15), textcoords='data',
                horizontalalignment='left', verticalalignment='top', fontsize=9)
    plot_sta(ax2, 0, t_AP, [sta_mean_good_APs], [sta_std_good_APs])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Mem. pot. (mV)')
    ax2.set_ylim(-75, 15)

    # ax3 = pl.subplot(outer[1, 4])
    # ax3.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    # plot_v_hist(ax3, 0, t_AP, bins_v, [v_hist])
    # ax3.set_xlabel('Time (ms)')
    # ax3.set_ylabel('Mem. pot. \ndistr. (mV)')
    # ax3.set_ylim(-75, 15)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAPs_in_vivo.png'))

    pl.show()