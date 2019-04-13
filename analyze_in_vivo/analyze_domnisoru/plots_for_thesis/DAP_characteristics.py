from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
pl.style.use('paper_subplots')

if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    cell_ids = get_cell_ids_DAP_cells()
    cell_type_dict = get_celltype_dict(save_dir)

    # parameters
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    cell_type = 'grid_cells'
    cell_ids_grid = np.array(load_cell_ids(save_dir, cell_type))
    DAP_cells_idxs = np.array([np.where(cell_id == cell_ids_grid)[0][0] for cell_id in cell_ids])
    DAP_deflection = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_deflection.npy'))
    DAP_amp = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_amp.npy'))
    DAP_width = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_width.npy'))
    DAP_time = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_time.npy'))

    # plot
    # fig = pl.figure(figsize=(3, 4))
    # n_rows, n_columns = 1, 2
    # outer = gridspec.GridSpec(n_rows, n_columns)

    # dap characteristics
    # characteristics = ['DAP deflection (mV)', 'DAP time (ms)']  # 'DAP amp. (mV)', 'DAP width (ms)']
    # characteristics_arrays = [DAP_deflection, DAP_time]  # DAP_amp, DAP_width]
    # for characteristic_idx, characteristic in enumerate(characteristics):
    #     ax = pl.Subplot(fig, outer[0, characteristic_idx])
    #     fig.add_subplot(ax)
    #
    #     plot_with_markers(ax, np.zeros(len(cell_ids)), characteristics_arrays[characteristic_idx][DAP_cells_idxs],
    #                       cell_ids, cell_type_dict,
    #                       theta_cells=load_cell_ids(save_dir, 'giant_theta'), DAP_cells=cell_ids, legend=False)
    #     ax.set_ylabel(characteristic)
    #     ax.set_xticks([])

    fig, ax = pl.subplots()
    handles = plot_with_markers(ax, DAP_time[DAP_cells_idxs], DAP_deflection[DAP_cells_idxs], cell_ids, cell_type_dict,
                      theta_cells=load_cell_ids(save_dir, 'giant_theta'), DAP_cells=cell_ids, legend=False)
    ax.set_xlabel('$Time_{AP-DAP}$ (ms)')
    ax.set_ylabel('DAP deflection (mV)')
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 2.5)
    pl.legend(handles=handles, loc='upper left')
    print cell_ids
    print 'DAP_deflections', DAP_deflection[DAP_cells_idxs]
    print 'DAP time', DAP_time[DAP_cells_idxs]
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'dap_characteristics.png'))
    pl.show()