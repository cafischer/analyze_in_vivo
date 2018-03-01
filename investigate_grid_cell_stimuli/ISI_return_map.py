from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load import load_full_runs
pl.style.use('paper')


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/ISI_hist'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]

        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5

        # ISI histogram
        ISIs_per_cell[i] = get_ISIs(v, t, AP_threshold)

        # save and plot
        save_dir_cell = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # 2d return
        pl.figure()
        pl.title(cell_id.split('_')[1], fontsize=16)
        pl.plot(ISIs_per_cell[i][:-1], ISIs_per_cell[i][1:], color='k', marker='o', linestyle='', markersize=6)
        pl.xlabel('ISI(n)')
        pl.ylabel('ISI(n+1)')
        pl.xlim(0, 200)
        pl.ylim(0, 200)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map.png'))
        #pl.show()

        # 3d return
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(cell_id.split('_')[1], fontsize=16)
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
        pl.show()

    # save and plot
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    pl.figure(figsize=(7.4, 4.8))
    for i, cell_id in enumerate(cell_ids):
        pl.plot(ISIs_per_cell[i][:-1], ISIs_per_cell[i][1:], label=cell_id.split('_')[1], color=colors[i],
                marker='o', linestyle='', markersize=6, alpha=0.6)
    pl.xlabel('ISI(n)')
    pl.ylabel('ISI(n+1)')
    pl.xlim(0, 200)
    pl.ylim(0, 200)
    pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pl.subplots_adjust(right=0.7, bottom=0.13, top=0.94)
    pl.savefig(os.path.join(save_dir, 'ISI_return_map.png'))
    pl.show()