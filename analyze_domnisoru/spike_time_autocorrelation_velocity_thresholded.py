from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs, get_cumulative_ISI_hist
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from analyze_in_vivo.analyze_domnisoru.spike_time_autocorrelation import get_autocorrelation, change_bin_size_of_spike_train
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from cell_characteristics import to_idx
import warnings
pl.style.use('paper')


def get_ISIs_for_groups(group_indicator, AP_max_idxs, t, max_ISI):
    starts, ends = get_starts_ends_group_of_ones(group_indicator.astype(int))
    ISIs_groups = []
    for start, end in zip(starts, ends):
        AP_max_idxs_inside = AP_max_idxs[np.logical_and(start < AP_max_idxs, AP_max_idxs < end)]

        ISIs = get_ISIs(AP_max_idxs_inside, t)
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_groups.extend(ISIs)
    return ISIs_groups


def get_auto_corr_for_group(group_indicator, spike_train, max_lag_idx, bin_size, dt):
    starts, ends = get_starts_ends_group_of_ones(group_indicator.astype(int))
    auto_corrs = np.zeros(len(starts), dtype=object)
    for group_idx, (start, end) in enumerate(zip(starts, ends)):
        spike_train_inside = spike_train[start:end + 1]
        spike_train_new = change_bin_size_of_spike_train(spike_train_inside, bin_size, dt)
        auto_corrs[group_idx] = get_autocorrelation(spike_train_new, max_lag_idx)
    # sum over auto-correlations and normalize
    auto_corr = np.sum(auto_corrs, 0)
    auto_corr[max_lag_idx] = 0  # for better plotting
    auto_corr /= (np.sum(auto_corr) * bin_size)  # normalize
    return auto_corr


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr/velocity_thresholding'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    max_lag = 250
    bin_size = 1.0  # ms
    max_lag_idx = to_idx(max_lag, bin_size)
    t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_size
    velocity_threshold = 1  # cm/sec

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    auto_corr_above_cells = np.zeros(len(cell_ids), dtype=object)
    auto_corr_under_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        AP_max_idxs = data['spiketimes']

        # get spike-time auto-correlation under and above velocity threshold
        above = velocity >= velocity_threshold
        under = velocity < velocity_threshold
        spike_train = get_spike_train(AP_max_idxs, len(v))

        auto_corr_above_cells[cell_idx] = get_auto_corr_for_group(above, spike_train, max_lag_idx, bin_size, dt)
        auto_corr_under_cells[cell_idx] = get_auto_corr_for_group(under, spike_train, max_lag_idx, bin_size, dt)


        # # plots
        # pl.figure()
        # pl.bar(t_auto_corr, auto_corr, bin_size, color='0.5', align='center')
        #
        # from analyze_in_vivo.analyze_domnisoru.plot_utils import find_most_equal_divisors
        # n_rows, n_columns = find_most_equal_divisors(len(starts))
        # fig, axes = pl.subplots(n_rows, n_columns, sharex='all', squeeze=True, figsize=(9, 9))
        # axes = axes.flatten()
        # for i in range(len(starts)):
        #     axes[i].bar(t_auto_corr, auto_corrs[i], bin_size, color='0.5', align='center')
        #     axes[i].set_xticks([])
        #     axes[i].set_yticks([])
        # pl.tight_layout()
        # pl.show()

    # plot
    def plot_auto_corr(ax, cell_idx, subplot_idx, t_auto_corr, auto_corr_above_cells, auto_corr_under_cells):
        if subplot_idx == 0:
            ax.bar(t_auto_corr, auto_corr_above_cells[cell_idx], bin_size, color='0.5', align='center')

            ax.set_xticks([])
            ax.set_xlabel('')
            ax.annotate('$\geq$ vel. thresh.', xy=(ax.get_xlim()[0], ax.get_ylim()[1]), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)
        if subplot_idx == 1:
            ax.bar(t_auto_corr, auto_corr_under_cells[cell_idx], bin_size, color='0.5', align='center')

            ax.annotate('$<$ vel. thresh.', xy=(ax.get_xlim()[0], ax.get_ylim()[1]), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)


    plot_kwargs = dict(t_auto_corr=t_auto_corr, auto_corr_above_cells=auto_corr_above_cells,
                       auto_corr_under_cells=auto_corr_under_cells)
    plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_auto_corr, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', n_subplots=2,
                            save_dir_img=os.path.join(save_dir_img, 'autocorr_lag_'+str(max_lag)+'.png'))
    pl.show()