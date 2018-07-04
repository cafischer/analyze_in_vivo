from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from cell_fitting.util import init_nan


def get_n_th_burst_indicator(short_ISI_indicator):
    n_th_burst_indicator = np.zeros(len(short_ISI_indicator))
    counter = 1
    for i in range(len(short_ISI_indicator)):
        if i > 0 and short_ISI_indicator[i] == 0 and short_ISI_indicator[i - 1] == 1:
            n_th_burst_indicator[i] = counter
            counter = 1
        elif short_ISI_indicator[i] == 1:
            n_th_burst_indicator[i] = counter
            counter += 1
        else:
            n_th_burst_indicator[i] = 0
    return n_th_burst_indicator


def plot_all_cells(cell_type_dict, burst_numbers, mean_v_n_th_burst_cells, std_v_n_th_burst_cells):
    if cell_type == 'grid_cells':
        def plot_fun(ax, cell_idx, burst_numbers, mean_v_n_th_burst_cells, std_v_n_th_burst_cells):
            ax.axhline(0, color='0.5', linestyle='--')
            ax.errorbar(burst_numbers, mean_v_n_th_burst_cells[cell_idx, :],
                        yerr=std_v_n_th_burst_cells[cell_idx, :], color='k', capsize=2)
            ax.set_xticks(burst_numbers)
            ax.set_xticklabels(burst_numbers)

        plot_kwargs = dict(burst_numbers=burst_numbers, mean_v_n_th_burst_cells=mean_v_n_th_burst_cells,
                           std_v_n_th_burst_cells=std_v_n_th_burst_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs,
                                xlabel='n-th AP \nin burst', ylabel='Difference to \n1st AP (mV)',
                                save_dir_img=os.path.join(save_dir_img, 'diff_1st_AP.png'))


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    bins = np.arange(1, 15 + 1, 1)
    ISI_burst = 10
    use_AP_max_idxs_domnisoru = True
    burst_numbers = np.array([2, 3, 4, 5])
    mean_v_n_th_burst_cells = init_nan((len(cell_ids), len(burst_numbers)))
    std_v_n_th_burst_cells = init_nan((len(cell_ids), len(burst_numbers)))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_th_burst_indicator = get_n_th_burst_indicator(short_ISI_indicator)

        v_APs = np.zeros(len(burst_numbers), dtype=object)
        t_APs = np.zeros(len(burst_numbers), dtype=object)
        n_APs = np.zeros(len(burst_numbers), dtype=object)
        for i, n in enumerate(burst_numbers):
            n_th_burst_idxs = np.where(n_th_burst_indicator == n)[0]
            mean_v_n_th_burst_cells[cell_idx, i] = np.mean(v[AP_max_idxs[n_th_burst_idxs - (n-1)]]
                                                           - v[AP_max_idxs[n_th_burst_idxs]])
            std_v_n_th_burst_cells[cell_idx, i] = np.std(v[AP_max_idxs[n_th_burst_idxs - (n-1)]]
                                                         - v[AP_max_idxs[n_th_burst_idxs]])

            # for testing
            n_APs[i] = min(len(n_th_burst_idxs), 5)
            idxs = range(len(n_th_burst_idxs))
            np.random.shuffle(idxs)
            v_APs[i] = []
            t_APs[i] = []
            for idx in range(n_APs[i]):
                idx_rand = idxs[idx]
                v_APs[i].append(v[AP_max_idxs[n_th_burst_idxs[idx_rand]-(n-1)]:AP_max_idxs[n_th_burst_idxs[idx_rand]]])
                t_APs[i].append(np.arange(len(v_APs[i][idx])) * dt)

        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        colors = pl.cm.get_cmap('jet')(np.linspace(0, 1, 5))
        fig, ax = pl.subplots(1, len(burst_numbers), figsize=(13, 5), sharey='all', sharex='all')
        for i in range(len(burst_numbers)):
            for idx in range(n_APs[i]):
                ax[i].set_title('Burst length: %i' % burst_numbers[i])
                ax[i].axhline(v_APs[i][idx][0], linestyle='--', color=colors[idx])
                ax[i].plot(t_APs[i][idx], v_APs[i][idx], color=colors[idx])
                ax[i].set_xlabel('Time (ms)')
                ax[i].set_ylabel('Mem. pot. (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'diff_1st_AP_samples.png'))
        #pl.show()

        pl.close('all')
        # pl.figure()
        # pl.errorbar(burst_numbers, mean_v_n_th_burst_cells[cell_idx, :],
        #             yerr=std_v_n_th_burst_cells[cell_idx, :], color='k')
        # pl.xlabel('n-th AP in burst')
        # pl.ylabel('Difference to 1st AP (mV)')
        # pl.xticks(burst_numbers, burst_numbers)
        # pl.tight_layout()
        # pl.show()

    # plot all cells
    pl.close('all')
    plot_all_cells(cell_type_dict, burst_numbers, mean_v_n_th_burst_cells, std_v_n_th_burst_cells)
    #pl.show()