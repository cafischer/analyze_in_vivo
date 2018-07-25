from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from cell_characteristics import to_idx
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.n_spikes_in_burst import get_n_spikes_in_burst
from cell_fitting.util import init_nan
pl.style.use('paper')


def get_avg_firing_rate_from_spike_trains(spike_train_APs, t_AP_binned, bins_t, bin_width_t):
    if spike_train_APs is None:
        return init_nan(len(bins_t)-1)
    sum_spike_train_APs = np.sum(spike_train_APs, 0) / np.size(spike_train_APs, 0)
    avg_firing_rate_cells = np.array([np.sum(sum_spike_train_APs[t_AP_binned == i])
                                                      for i in range(len(bins_t) - 1)]) / bin_width_t * 1000.0
    avg_firing_rate_cells[bins_t[:-1] == 0] = 0  # for plotting
    return avg_firing_rate_cells


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/Harris/sta_firing_rate'
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
    before_AP = 200  # ms
    after_AP = 200  # ms
    bin_width_t = 2  # ms
    bins_t = np.arange(-before_AP, after_AP + 2 * bin_width_t, bin_width_t)

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    avg_firing_rate_burst_cells = [0] * len(cell_ids)
    avg_firing_rate_3sburst_cells = [0] * len(cell_ids)
    avg_firing_rate_g5sburst_cells = [0] * len(cell_ids)
    avg_firing_rate_single_cells = [0] * len(cell_ids)
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
        spike_train = get_spike_train(AP_max_idxs, len(v))
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        starts_burst, ends_burst = get_starts_ends_group_of_ones(short_ISI_indicator.astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst + 1],
                                             AP_max_idxs[~short_ISI_indicator]))
        n_spikes_in_bursts = get_n_spikes_in_burst(short_ISI_indicator.astype(int))
        AP_max_idxs_3sburst = AP_max_idxs_burst[n_spikes_in_bursts == 3]
        AP_max_idxs_g5sburst = AP_max_idxs_burst[n_spikes_in_bursts >= 5]

        # get spike trains for all APs
        spike_train_APs_burst = find_all_AP_traces(spike_train, before_AP_idx, after_AP_idx, AP_max_idxs_burst)
        spike_train_APs_3sburst = find_all_AP_traces(spike_train, before_AP_idx, after_AP_idx, AP_max_idxs_3sburst)
        spike_train_APs_g5sburst = find_all_AP_traces(spike_train, before_AP_idx, after_AP_idx, AP_max_idxs_g5sburst)
        spike_train_APs_single = find_all_AP_traces(spike_train, before_AP_idx, after_AP_idx, AP_max_idxs_single)

        # average
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt - before_AP
        t_AP_binned = np.digitize(t_AP, bins_t)-1
        avg_firing_rate_burst_cells[cell_idx] = get_avg_firing_rate_from_spike_trains(spike_train_APs_burst,
                                                                                      t_AP_binned, bins_t, bin_width_t)
        avg_firing_rate_3sburst_cells[cell_idx] = get_avg_firing_rate_from_spike_trains(spike_train_APs_3sburst,
                                                                                      t_AP_binned, bins_t, bin_width_t)
        avg_firing_rate_g5sburst_cells[cell_idx] = get_avg_firing_rate_from_spike_trains(spike_train_APs_g5sburst,
                                                                                      t_AP_binned, bins_t, bin_width_t)
        avg_firing_rate_single_cells[cell_idx] = get_avg_firing_rate_from_spike_trains(spike_train_APs_single,
                                                                                      t_AP_binned, bins_t, bin_width_t)

        # save and plot
        # save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)

        # pl.figure()
        # pl.title('Burst')
        # pl.bar(bins_t[:-1], avg_firing_rate_burst_cells[cell_idx], width=bin_width_t, color='k', align='edge')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Firing rate (Hz)')
        # pl.tight_layout()
        # #pl.show()

        # pl.figure()
        # pl.title('Single')
        # pl.bar(bins_t[:-1], avg_firing_rate_single_cells[cell_idx], width=bin_width_t, color='k', align='edge')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Firing rate (Hz)')
        # pl.tight_layout()
        # pl.show()

    # save and plot
    def plot_sta(ax, cell_idx, bins_t, avg_firing_rate_cells, bin_width_t):
        ax.bar(bins_t[:-1], avg_firing_rate_cells[cell_idx], width=bin_width_t, color='k', align='edge')

    if cell_type == 'grid_cells':
        plot_kwargs = dict(bins_t=bins_t, avg_firing_rate_cells=avg_firing_rate_burst_cells,
                           bin_width_t=bin_width_t)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Firing rate (Hz)',
                                save_dir_img=os.path.join(save_dir_img, 'sta_firing_rate_burst.png'))

        plot_kwargs = dict(bins_t=bins_t, avg_firing_rate_cells=avg_firing_rate_3sburst_cells,
                           bin_width_t=bin_width_t)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Firing rate (Hz)',
                                save_dir_img=os.path.join(save_dir_img, 'sta_firing_rate_3sburst.png'))

        plot_kwargs = dict(bins_t=bins_t, avg_firing_rate_cells=avg_firing_rate_g5sburst_cells,
                           bin_width_t=bin_width_t)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Firing rate (Hz)',
                                save_dir_img=os.path.join(save_dir_img, 'sta_firing_rate_g5sburst.png'))

        plot_kwargs = dict(bins_t=bins_t, avg_firing_rate_cells=avg_firing_rate_single_cells,
                           bin_width_t=bin_width_t)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Firing rate (Hz)',
                                save_dir_img=os.path.join(save_dir_img, 'sta_firing_rate_single.png'))
        pl.show()