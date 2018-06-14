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
    avg_firing_rate_burst_cells = [0] * len(cell_ids)
    std_firing_rate_burst_cells = [0] * len(cell_ids)
    avg_firing_rate_single_cells = [0] * len(cell_ids)
    std_firing_rate_single_cells = [0] * len(cell_ids)
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

        # identify single and burst APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        starts_burst, ends_burst = get_start_end_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))

        # running firing rate
        firing_rate = init_nan(len(t))
        event_rate = init_nan(len(t))
        fraction_burst = init_nan(len(t))
        for i, t_step in enumerate(t):
            idx = np.logical_and(i - window_size_idx/2.0 <= AP_max_idxs, AP_max_idxs <= i + window_size_idx/2.0)
            idx_single = np.logical_and(i - window_size_idx / 2.0 <= AP_max_idxs_single,
                                        AP_max_idxs_single <= i + window_size_idx / 2.0)
            idx_burst = np.logical_and(i - window_size_idx / 2.0 <= AP_max_idxs_burst,
                                       AP_max_idxs_burst <= i + window_size_idx / 2.0)
            firing_rate[i] = len(AP_max_idxs[idx]) / (window_size/1000.0)
            n_events = (len(AP_max_idxs_single[idx_single])+len(AP_max_idxs_burst[idx_burst]))
            event_rate[i] = n_events / (window_size/1000.0)
            if n_events != 0:
                fraction_burst[i] = len(AP_max_idxs_burst[idx_burst]) / n_events

        # get firing rate for all APs
        firing_rate_APs_burst = find_all_AP_traces(firing_rate, before_AP_idx, after_AP_idx, AP_max_idxs_burst)
        firing_rate_APs_single = find_all_AP_traces(firing_rate, before_AP_idx, after_AP_idx, AP_max_idxs_single)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt - before_AP

        # average
        avg_firing_rate_burst_cells.append(np.mean(firing_rate_APs_burst, 0))
        std_firing_rate_burst_cells.append(np.std(firing_rate_APs_burst, 0))
        avg_firing_rate_single_cells.append(np.mean(firing_rate_APs_single, 0))
        std_firing_rate_single_cells.append(np.std(firing_rate_APs_single, 0))

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
        # pl.figure()
        # pl.title(cell_id, fontsize=16)
        # pl.plot(t, firing_rate, 'k')
        # pl.plot(t, event_rate, 'b')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Firing rate (Hz)')
        # pl.tight_layout()
        # pl.show()
        #
        # pl.figure()
        # pl.plot(t_AP, avg_firing_rate, 'k')
        # pl.fill_between(t_AP, avg_firing_rate - std_firing_rate, avg_firing_rate + std_firing_rate, color='k',
        #                 alpha=0.5)
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Firing rate (Hz)')
        # pl.tight_layout()
        # pl.show()
        bin_width_event = 2
        steps = np.arange(0, 50, 1)
        mean_frac = np.zeros(len(steps))
        std_frac= np.zeros(len(steps))
        for i, er in enumerate(steps):
            idx = np.logical_and(er - bin_width_event/2.0 <= event_rate,  event_rate <= er + bin_width_event/2.0)
            mean_frac[i] = np.nanmean(fraction_burst[idx])
            std_frac[i] = np.nanstd(fraction_burst[idx])
        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(event_rate, fraction_burst, 'o', color='0.5', markersize=4)
        pl.plot(steps, mean_frac, 'r')
        pl.fill_between(steps, mean_frac - std_frac, mean_frac + std_frac, color='r', alpha=0.5)
        pl.xlabel('Event rate (Hz)')
        pl.ylabel('Fraction burst events')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'event_rate_vs_fraction_burst.png'))
        pl.show()

    # save and plot
    pl.close('all')
    if cell_type == 'grid_cells':
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

                    axes[i1, i2].plot(t_AP, avg_firing_rate_burst_cells[cell_idx], 'k')
                    axes[i1, i2].fill_between(t_AP, avg_firing_rate_burst_cells[cell_idx] - std_firing_rate_burst_cells[cell_idx],
                                              avg_firing_rate_burst_cells[cell_idx] + std_firing_rate_burst_cells[cell_idx],
                                              color='k', alpha=0.5)

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
        pl.savefig(os.path.join(save_dir_img, cell_type, 'sta_firing_rate_burst.png'))

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

                    axes[i1, i2].plot(t_AP, avg_firing_rate_single_cells[cell_idx], 'k')
                    axes[i1, i2].fill_between(t_AP, avg_firing_rate_single_cells[cell_idx] - std_firing_rate_single_cells[cell_idx],
                                              avg_firing_rate_single_cells[cell_idx] + std_firing_rate_single_cells[cell_idx],
                                              color='k', alpha=0.5)

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
        pl.savefig(os.path.join(save_dir_img, cell_type, 'sta_firing_rate_single.png'))
        pl.show()