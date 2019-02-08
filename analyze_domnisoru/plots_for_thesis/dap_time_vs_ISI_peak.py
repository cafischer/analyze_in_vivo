import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, get_celltype_dict, load_cell_ids, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
pl.style.use('paper_subplots')


if __name__ == '__main__':
    #save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'

    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    save_dir_DAP_times = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/not_detrended/all/grid_cells'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type_dict = get_celltype_dict(save_dir)
    max_ISI = 200
    bin_width = 2.0  # ms
    save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_' + str(max_ISI), 'grid_cells')
    save_dir_ISI_hist = os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_' + str(max_ISI))

    # load stuff
    grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in grid_cells])
    # DAP_time = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_time.npy')) TODO
    DAP_time = np.load(os.path.join(save_dir_DAP_times, 'DAP_times.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'grid_cells', 'peak_ISI_hist_'+str(max_ISI)+'_'+str(bin_width)+'.npy'))
    peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak

    # # plot correlation DAP-time and peak ISI-hist
    # fig, ax = pl.subplots()
    # cell_idx = len(cell_ids)-1
    #
    # ax.plot(np.arange(0, 10), np.arange(0, 10), '0.5', linestyle='--')
    # #ax.fill_between(np.arange(0, 10), np.arange(0, 10)-1, np.arange(0, 10)+1, color='0.7')
    # plot_with_markers(ax, DAP_time, peak_ISI_hist, grid_cells, cell_type_dict,
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # ax.set_xlim(0, 7)
    # ax.set_ylim(0, 7)
    # ax.set_xticks(np.arange(0, 8, 2))
    # ax.set_yticks(np.arange(0, 8, 2))
    # ax.set_aspect('equal', adjustable='box-forced')
    # ax.set_ylabel('Peak of ISI hist. (ms)')
    # ax.set_xlabel('DAP time (ms)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'dap_time_vs_ISI_peak.png'))
    # pl.show()

    # # alternative figure
    # f, (ax, ax2) = pl.subplots(2, 1,  gridspec_kw={'height_ratios': [1, 5]}, figsize=(6, 6))
    #
    # # plot the same data on both axes
    # ax.plot(np.arange(0, 20), np.arange(0, 20), '0.5', linestyle='--')
    # plot_with_markers(ax, DAP_time[burst_label], peak_ISI_hist[burst_label], grid_cells[burst_label], cell_type_dict,
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional,
    #                   edgecolor='r', legend=False)
    # plot_with_markers(ax, DAP_time[~burst_label], peak_ISI_hist[~burst_label], grid_cells[~burst_label], cell_type_dict,
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional,
    #                   edgecolor='b', legend=False)
    #
    # ax2.plot(np.arange(0, 20), np.arange(0, 20), '0.5', linestyle='--')
    # handles = plot_with_markers(ax2, DAP_time[burst_label], peak_ISI_hist[burst_label], grid_cells[burst_label],
    #                             cell_type_dict, theta_cells=theta_cells, DAP_cells=DAP_cells,
    #                             DAP_cells_additional=DAP_cells_additional, edgecolor='r', legend=False)
    # plot_with_markers(ax2, DAP_time[~burst_label], peak_ISI_hist[~burst_label], grid_cells[~burst_label], cell_type_dict,
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional,
    #                   edgecolor='b', legend=False)
    # handles += [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    # ax2.legend(handles=handles, loc='lower right')
    # ax2.set_ylabel('Peak of ISI hist. (ms)')
    # ax2.set_xlabel('Time$_{AP-DAP}$ (ms)')
    #
    # # zoom-in / limit the view to different portions of the data
    # ax.set_xlim(0, 9)
    # ax2.set_xlim(0, 9)
    # ax.set_ylim(33.5, 36.5)  # outliers only
    # ax2.set_ylim(0, 9)  # most of the data
    #
    # # hide the spines between ax and ax2
    # ax.spines['bottom'].set_visible(False)
    # ax.set_yticks([34, 36])
    # ax.set_xticklabels([])
    # ax.set_xticks([])
    # ax2.set_yticks(np.arange(0, 10, 2))
    # ax2.set_xticks(ax2.get_yticks())
    # ax2.set_xticklabels(ax2.get_yticks())
    #
    #
    # # cut diagonal lines
    # d = .015  # how big to make the diagonal lines in axes coordinates
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax2.plot((-d, +d), (-d-0.125, +d-0.125), **kwargs)        # top-left diagonal
    # ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
    #
    # pl.tight_layout()
    # pl.subplots_adjust(hspace=0.04)
    # pl.savefig(os.path.join(save_dir_img, 'dap_time_vs_ISI_peak_'+str(max_ISI)+'_'+str(bin_width)+'.png'))

    # for Andreas:
    f, ax = pl.subplots()

    # plot the same data on both axes
    ax.plot(np.arange(0, 20), np.arange(0, 20), '0.5', linestyle='--')
    ax.fill_between(np.arange(0, 20), np.arange(0, 20)-bin_width, np.arange(0, 20)+bin_width, color='0.5', alpha=0.15)
    plot_with_markers(ax, DAP_time[burst_label], peak_ISI_hist[burst_label], grid_cells[burst_label], cell_type_dict,
                      theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional,
                      edgecolor='r', legend=False)
    plot_with_markers(ax, DAP_time[~burst_label], peak_ISI_hist[~burst_label], grid_cells[~burst_label], cell_type_dict,
                      theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional,
                      edgecolor='b', legend=False)
    ax.set_ylabel('Peak of ISI hist. (ms)')
    ax.set_xlabel('Time$_{AP-DAP}$ (ms)')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    for i in range(len(grid_cells)):
        if grid_cells[i] == 's79_0003' or grid_cells[i] == 's109_0002':
            ax.annotate(grid_cells[i], xy=(DAP_time[i] + 0.05, peak_ISI_hist[i] + 1.0), fontsize=7)
        else:
            ax.annotate(grid_cells[i], xy=(DAP_time[i]+0.15, peak_ISI_hist[i]+0.2), fontsize=7)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'dap_time_vs_ISI_peak_'+str(max_ISI)+'_'+str(bin_width)+'.png'))
    pl.show()
