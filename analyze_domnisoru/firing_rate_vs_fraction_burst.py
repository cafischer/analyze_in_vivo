from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')


def plot_rate_vs_burst(ax, cell_idx, rate_cells, fraction_burst_cells):
    # ax.plot(rate_cells[cell_idx], fraction_burst_cells[cell_idx], 'o', color='0.5', markersize=3)  # can cause memory error
    window_rate = 2
    steps = np.arange(0, np.max(rate_cells[cell_idx]), 2)
    mean_frac = np.zeros(len(steps))
    std_frac = np.zeros(len(steps))
    for i, er in enumerate(steps):
        idx = np.logical_and(er - window_rate / 2.0 <= rate_cells[cell_idx],
                             rate_cells[cell_idx] <= er + window_rate / 2.0)
        mean_frac[i] = np.nanmean(fraction_burst_cells[cell_idx][idx])
        std_frac[i] = np.nanstd(fraction_burst_cells[cell_idx][idx])
    ax.errorbar(steps, mean_frac, yerr=std_frac, capsize=2, color='k')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/Harris/firing_rate_vs_fraction_burst'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    ISI_burst = 8  # ms
    window_size = 2000  # ms

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    fraction_burst_cells = [0] * len(cell_ids)
    event_rate_cells = [0] * len(cell_ids)
    firing_rate_cells = [0] * len(cell_ids)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        window_size_idx = to_idx(window_size, dt)

        # identify single and burst APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        spike_train = get_spike_train(AP_max_idxs, len(v))
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        starts_burst, ends_burst = get_start_end_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))

        # running firing rate
        firing_rate = init_nan(len(t))
        event_rate = init_nan(len(t))
        fraction_burst = np.zeros(len(t))

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

        fraction_burst_cells[cell_idx] = fraction_burst
        event_rate_cells[cell_idx] = event_rate
        firing_rate_cells[cell_idx] = firing_rate

        # slower implementation due to many zeros
        # spike_indicator = np.zeros(len(t), dtype=int)
        # spike_indicator[AP_max_idxs] = 1
        # spike_indicator[AP_max_idxs_burst] = 2
        # spike_indicator[AP_max_idxs_single] = 3
        #
        # half_window_size = int(np.round(window_size_idx / 2.0))
        #
        # for i, t_step in enumerate(t):
        #     window = spike_indicator[max(i - half_window_size, 0): i + 1 + half_window_size]
        #     counts = np.bincount(window, minlength=4)
        #
        #     firing_rate[i] = (counts[1] + counts[2] + counts[3]) / (window_size/1000.0)
        #     n_events = counts[2] + counts[3]
        #     event_rate[i] = n_events / (window_size/1000.0)
        #     if n_events != 0:
        #         fraction_burst[i] = float(counts[2]) / n_events

        # save and plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)

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

        # fig, ax = pl.subplots()
        # ax.set_title(cell_id, fontsize=16)
        # plot_rate_vs_burst(ax, cell_idx, event_rate_cells, fraction_burst_cells)
        # pl.xlabel('Event rate (Hz)')
        # pl.ylabel('Fraction burst events')
        # pl.tight_layout()
        # #pl.savefig(os.path.join(save_dir_cell, 'event_rate_vs_fraction_burst.png'))
        # pl.show()

    # save and plot
    if cell_type == 'grid_cells':
        plot_kwargs = dict(rate_cells=event_rate_cells, fraction_burst_cells=fraction_burst_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_rate_vs_burst, plot_kwargs,
                                xlabel='Event \nrate (Hz)', ylabel='Fraction burst events',
                                save_dir_img=os.path.join(save_dir_img, 'event_rate_vs_fraction_burst.png'))

        plot_kwargs = dict(rate_cells=firing_rate_cells, fraction_burst_cells=fraction_burst_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_rate_vs_burst, plot_kwargs,
                                xlabel='Firing \nrate (Hz)', ylabel='Fraction burst events',
                                save_dir_img=os.path.join(save_dir_img, 'firing_rate_vs_fraction_burst.png'))
        pl.show()