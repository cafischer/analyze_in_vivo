import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, get_celltype_dict, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = get_cell_ids_DAP_cells()

    # plot correlation DAP-time and peak ISI-hist
    fig, ax = pl.subplots()
    cell_idx = len(cell_ids)-1
    grid_cells = load_cell_ids(save_dir, 'grid_cells')
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells()
    DAP_time = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_time.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'grid_cells', 'peak_ISI_hist.npy'))
    peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak

    ax.plot(np.arange(0, 10), np.arange(0, 10), '0.5', linestyle='--')
    #ax.fill_between(np.arange(0, 10), np.arange(0, 10)-1, np.arange(0, 10)+1, color='0.7')
    plot_with_markers(ax, DAP_time, peak_ISI_hist, grid_cells, cell_type_dict,
                      theta_cells=theta_cells, DAP_cells=DAP_cells)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_xticks(np.arange(0, 8, 2))
    ax.set_yticks(np.arange(0, 8, 2))
    ax.set_aspect('equal', adjustable='box-forced')
    ax.set_ylabel('Peak of ISI hist. (ms)')
    ax.set_xlabel('DAP time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'dap_time_vs_ISI_peak.png'))
    pl.show()
