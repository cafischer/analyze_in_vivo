from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'pyramidal_layer2'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    filter_long_ISIs = True
    filter_long_ISIs_max = 200
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(filter_long_ISIs_max))

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt, interval=2, v_diff_onset_max=5)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= filter_long_ISIs_max]
        n_ISIs[i] = len(ISIs)
        ISIs_per_cell[i] = ISIs

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # 2d return
        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(ISIs_per_cell[i][:-1], ISIs_per_cell[i][1:], color='k', marker='o', linestyle='', markersize=6)
        pl.xlabel('ISI(n)')
        pl.ylabel('ISI(n+1)')
        pl.xlim(0, 200)
        pl.ylim(0, 200)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map.png'))

        # 3d return
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(cell_id, fontsize=16)
        ax.scatter(ISIs_per_cell[i][:-2], ISIs_per_cell[i][1:-1], ISIs_per_cell[i][2:],
                   color='k', marker='o') #, markersize=6)
        ax.set_xlabel('ISI(n)', fontsize=16)
        ax.set_ylabel('ISI(n+1)', fontsize=16)
        ax.set_zlabel('ISI(n+2)', fontsize=16)
        ax.set_xlim3d(0, 200)
        ax.set_ylim3d(200, 0)
        ax.set_zlim3d(0, 200)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map_3d.png'))
        #pl.show()

    # save and plot
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    pl.figure(figsize=(7.4, 4.8))
    for i, cell_id in enumerate(cell_ids):
        pl.plot(ISIs_per_cell[i][:-1], ISIs_per_cell[i][1:], label=cell_id, color=colors[i],
                marker='o', linestyle='', markersize=6, alpha=0.6)
    pl.xlabel('ISI(n)')
    pl.ylabel('ISI(n+1)')
    pl.xlim(0, 200)
    pl.ylim(0, 200)
    pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pl.subplots_adjust(right=0.7, bottom=0.13, top=0.94)
    pl.savefig(os.path.join(save_dir_img, cell_type, 'ISI_return_map.png'))
    #pl.show()