from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_return_map
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
pl.style.use('paper')


if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    max_ISI = None  # None if you want to take all ISIs
    max_ISI_plot = 200  # ms
    burst_ISI = 8  # ms

    folder = 'max_ISI_' + str(max_ISI)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_cells = np.zeros(len(cell_ids), dtype=object)
    fraction_ISI_or_ISI_next_burst = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # ISIs
        ISIs = get_ISIs(AP_max_idxs, t)
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_cells[cell_idx] = ISIs
        fraction_ISI_or_ISI_next_burst[cell_idx] = float(sum(np.logical_or(ISIs[:-1] < burst_ISI,
                                                                           ISIs[1:] < burst_ISI))) / len(ISIs[1:])

        # save and plot
        # save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        #
        # # 2d return
        # pl.figure()
        # pl.title(cell_id, fontsize=16)
        # pl.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5', marker='o',
        #         linestyle='', markersize=3)
        # pl.xlabel('ISI[n] (ms)')
        # pl.ylabel('ISI[n+1] (ms)')
        # pl.xlim(0, max_ISI)
        # pl.ylim(0, max_ISI)
        # pl.legend()
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map.png'))
        # #pl.show()
        #
        # # 3d return
        # fig = pl.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_title(cell_id, fontsize=16)
        # ax.scatter(ISIs_cells[cell_idx][:-2], ISIs_cells[cell_idx][1:-1], ISIs_cells[cell_idx][2:],
        #            color='k', marker='o') #, markersize=6)
        # ax.set_xlabel('ISI[n] (ms)', fontsize=16)
        # ax.set_ylabel('ISI[n+1] (ms)', fontsize=16)
        # ax.set_zlabel('ISI[n+2] (ms)', fontsize=16)
        # ax.set_xlim3d(0, max_ISI)
        # ax.set_ylim3d(max_ISI, 0)
        # ax.set_zlim3d(0, max_ISI)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map_3d.png'))
        # #pl.show()
        # pl.close('all')

    # save and plot
    np.save(os.path.join(save_dir_img, 'fraction_ISI_or_ISI_next_burst.npy'), fraction_ISI_or_ISI_next_burst)

    if cell_type == 'grid_cells':
        burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
        colors_marker = np.zeros(len(burst_label), dtype=str)
        colors_marker[burst_label] = 'r'
        colors_marker[~burst_label] = 'b'

        params = {'legend.fontsize': 9}
        pl.rcParams.update(params)

        # plot return maps
        plot_kwargs = dict(ISIs_cells=ISIs_cells, max_ISI=max_ISI_plot)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)',
                                save_dir_img=os.path.join(save_dir_img, 'ISI_return_map.png'))
        #plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
        #                        xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', colors_marker=colors_marker,
        #                        wspace=0.18, save_dir_img=os.path.join(save_dir_img2, 'ISI_return_map.png'))

        plot_kwargs = dict(ISIs_cells=ISIs_cells, max_ISI=max_ISI, log_scale=True)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', colors_marker=colors_marker,
                                save_dir_img=os.path.join(save_dir_img, 'return_map_log_scale.png'))
        pl.show()