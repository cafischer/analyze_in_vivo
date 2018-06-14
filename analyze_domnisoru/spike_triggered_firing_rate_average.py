from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    ISI_burst = 8  # ms
    window_size = 1000  # ms
    before_AP = 200  # ms
    after_AP = 200  # ms

    # over cells
    firing_rate_cells = [0] * len(cell_ids)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        window_size_idx = to_idx(window_size, dt)
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

       # running firing rate
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        firing_rate = init_nan(len(t))
        for i, t_step in enumerate(t):
            idx = np.logical_and(i - window_size_idx/2.0 <= AP_max_idxs, AP_max_idxs <= i + window_size_idx/2.0)
            firing_rate[i] = len(AP_max_idxs[idx]) / (window_size/1000.0)
        firing_rate_cells.append(firing_rate)

        # identify single and burst APs
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        starts_burst, ends_burst = get_start_end_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))

        # get firing rate for all APs
        firing_rate_APs = find_all_AP_traces(firing_rate, before_AP_idx, after_AP_idx, AP_max_idxs_burst)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt - before_AP

        # average
        avg_firing_rate = np.mean(firing_rate_APs, 0)
        std_firing_rate = np.std(firing_rate_APs, 0)

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # pl.figure()
        # pl.title(cell_id, fontsize=16)
        # pl.plot(t, v, 'k')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Mem.Pot. (mV)')
        # pl.tight_layout()
        #
        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(t, firing_rate, 'k')
        pl.xlabel('Time (ms)')
        pl.ylabel('Firing rate (Hz)')
        pl.tight_layout()
        pl.show()

        # pl.figure()
        # for fr in firing_rate_APs:
        #     pl.plot(t_AP, fr)
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Firing rate (Hz)')
        # pl.tight_layout()

        pl.figure()
        pl.plot(t_AP, avg_firing_rate, 'k')
        pl.fill_between(t_AP, avg_firing_rate - std_firing_rate, avg_firing_rate + std_firing_rate, color='k',
                        alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Firing rate (Hz)')
        pl.tight_layout()
        pl.show()


    # save and plot
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
        pl.savefig(os.path.join(save_dir_img, cell_type, 'sta_firing_rate.png'))
        #pl.show()