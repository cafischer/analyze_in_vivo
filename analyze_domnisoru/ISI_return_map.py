from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_fitting.util import init_nan
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = True
    max_ISI = 200
    ISI_burst = 8  # ms
    steps = np.arange(0, max_ISI + 1.0, 1.0)
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_' + str(max_ISI))

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    median_cells = init_nan((len(cell_ids), len(steps)))
    prob_next_ISI_short = init_nan((len(cell_ids), len(steps)))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[cell_idx] = len(ISIs)
        ISIs_per_cell[cell_idx] = ISIs

        # running median
        bin_size = 5.0  # ms
        prev_ISI = ISIs_per_cell[cell_idx][:-1]
        next_ISI = ISIs_per_cell[cell_idx][1:]
        median = init_nan(len(steps))
        mean = init_nan(len(steps))
        for i, s in enumerate(steps):
            idx = np.logical_and(s - bin_size/2.0 <= prev_ISI, prev_ISI <= s + bin_size/2.0)
            median_cells[cell_idx, i] = np.median(next_ISI[idx])
            mean[i] = np.mean(next_ISI[idx])

            # probability next ISI < x ms
            prob_next_ISI_short[cell_idx, i] = np.sum(next_ISI[idx] < ISI_burst) / float(np.sum(idx))

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # 2d return
        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5', marker='o',
                linestyle='', markersize=3)
        pl.plot(steps, median_cells[cell_idx, :], 'k', label='median')
        #pl.plot(steps, mean, 'b')
        pl.xlabel('ISI[n] (ms)')
        pl.ylabel('ISI[n+1] (ms)')
        pl.xlim(0, max_ISI)
        pl.ylim(0, max_ISI)
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map.png'))
        #pl.show()

        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(steps, prob_next_ISI_short[cell_idx, :], 'k')
        pl.xlabel('ISI[n] (ms)')
        pl.ylabel('Prob. ISI[n+1] < %i ms' % ISI_burst)
        pl.tight_layout()
        #pl.show()

        # 3d return
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(cell_id, fontsize=16)
        ax.scatter(ISIs_per_cell[cell_idx][:-2], ISIs_per_cell[cell_idx][1:-1], ISIs_per_cell[cell_idx][2:],
                   color='k', marker='o') #, markersize=6)
        ax.set_xlabel('ISI[n] (ms)', fontsize=16)
        ax.set_ylabel('ISI[n+1] (ms)', fontsize=16)
        ax.set_zlabel('ISI[n+2] (ms)', fontsize=16)
        ax.set_xlim3d(0, max_ISI)
        ax.set_ylim3d(max_ISI, 0)
        ax.set_zlim3d(0, max_ISI)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map_3d.png'))
        #pl.show()
        pl.close('all')

    # save and plot
    # cm = pl.cm.get_cmap('plasma')
    # colors = cm(np.linspace(0, 1, len(cell_ids)))
    # pl.figure(figsize=(7.4, 4.8))
    # for i, cell_id in enumerate(cell_ids):
    #     pl.plot(ISIs_per_cell[i][:-1], ISIs_per_cell[i][1:], label=cell_id, color=colors[i],
    #             marker='o', linestyle='', markersize=6, alpha=0.6)
    # pl.xlabel('ISI(n)')
    # pl.ylabel('ISI(n+1)')
    # pl.xlim(0, 200)
    # pl.ylim(0, 200)
    # pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # pl.subplots_adjust(right=0.7, bottom=0.13, top=0.94)
    # pl.savefig(os.path.join(save_dir_img, cell_type, 'ISI_return_map.png'))
    # #pl.show()

    pl.close('all')
    if cell_type == 'grid_cells':
        # plot all return maps
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                    axes[i1, i2].plot(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5',
                                      marker='o', linestyle='', markersize=1, alpha=0.5)
                    axes[i1, i2].set_xlim(0, max_ISI)
                    axes[i1, i2].set_ylim(0, max_ISI)
                    axes[i1, i2].set_aspect('equal', adjustable='box-forced')
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('ISI[n] (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('ISI[n+1] (ms)')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.subplots_adjust(wspace=0.25)
        pl.savefig(os.path.join(save_dir_img, cell_type, 'return_map.png'))

        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                    axes[i1, i2].plot(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5',
                                      marker='o', linestyle='', markersize=1, alpha=0.5)
                    axes[i1, i2].plot(steps, median_cells[cell_idx, :], 'k', label='median')
                    axes[i1, i2].set_xlim(0, max_ISI)
                    axes[i1, i2].set_ylim(0, max_ISI)
                    axes[i1, i2].set_aspect('equal', adjustable='box-forced')
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('ISI[n] (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('ISI[n+1] (ms)')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.subplots_adjust(wspace=0.25)
        pl.savefig(os.path.join(save_dir_img, cell_type, 'return_map_with_median.png'))
        #pl.show()

        # plot prob. next ISI < x ms
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                    axes[i1, i2].plot(steps, prob_next_ISI_short[cell_idx, :], 'k')

                    axes[i1, i2].set_xlim(0, max_ISI)
                    axes[i1, i2].set_ylim(0, 1)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('ISI[n] (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Prob. ISI[n+1] < %i ms' % ISI_burst)
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.subplots_adjust(wspace=0.25)
        pl.savefig(os.path.join(save_dir_img, cell_type, 'prob_next_ISI.png'))
        pl.show()