from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_cell_group_grid
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from analyze_in_vivo.analyze_domnisoru.spike_time_autocorrelation import auto_correlate, change_bin_size_of_spike_train
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'DAP_cells'
    cell_ids = get_cell_ids_DAP_cells()
    param_list = ['Vm_ljpc', 'spiketimes']
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = True
    max_ISI = 200
    max_lag = 50
    burst_ISI = 8  # ms
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(max_ISI))
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # parameter
    bin_width = 1.0
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    # over cells
    ISIs_cells = np.zeros(len(cell_ids), dtype=object)
    ISI_hist = np.zeros((len(cell_ids), len(bins)-1))
    cum_ISI_hist_y = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_x = np.zeros(len(cell_ids), dtype=object)
    auto_corr_cells = np.zeros(len(cell_ids), dtype=object)

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

        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_cells[cell_idx] = ISIs

        # ISI histograms
        ISI_hist[cell_idx, :] = get_ISI_hist(ISIs, bins)
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs)

        # auto-correlation
        spike_train = get_spike_train(AP_max_idxs, len(v))  # for norm to firing rate: spike_train / len(AP_max_idxs)
        spike_train_new = change_bin_size_of_spike_train(spike_train, bin_width, dt)
        max_lag_idx = to_idx(max_lag, bin_width)

        auto_corr = auto_correlate(spike_train_new, max_lag_idx)
        auto_corr[max_lag_idx] = 0  # for better plotting
        auto_corr /= (np.sum(auto_corr) * bin_width)  # normalize
        auto_corr_cells[cell_idx] = auto_corr


    # plot all ISI hists
    def plot_ISI_hist(ax, cell_idx, subplot_idx, ISI_hist, cum_ISI_hist_x, cum_ISI_hist_y, ISIs_cells, max_ISI,
                      auto_corr_cells, max_lag, bin_width):
        if subplot_idx == 0:
            ax.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
                   bins[1] - bins[0], color='0.5', align='edge')
            cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[cell_idx], len(cum_ISI_hist_x[cell_idx]), max_ISI)
            cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[cell_idx], len(cum_ISI_hist_y[cell_idx]), 1.0)
            ax_twin = ax.twinx()
            ax_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
            ax_twin.set_xlim(0, max_ISI)
            ax_twin.set_ylim(0, 1)
            ax_twin.set_yticks([0, 1])
            ax.spines['right'].set_visible(True)
            ax.set_ylabel('Rel. frequency')
            ax.set_xlabel('ISI (ms)')
        elif subplot_idx == 1:
            ax.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
                    marker='o', linestyle='', markersize=1, alpha=0.5)
            ax.set_xlim(0, max_ISI)
            ax.set_ylim(0, max_ISI)
            ax.set_aspect('equal', adjustable='box-forced')
            ax.set_ylabel('ISI[n+1] (ms)')
            ax.set_xlabel('ISI[n] (ms)')
        elif subplot_idx == 2:
            max_lag_idx = to_idx(max_lag, bin_width)
            t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_width
            ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_width, color='0.5', align='center')
            ax.set_xlim(-max_lag, max_lag)
            ax.set_ylabel('Spike-time \nautocorrelation')
            ax.set_xlabel('Lag (ms)')


    plot_kwargs = dict(ISI_hist=ISI_hist, cum_ISI_hist_x=cum_ISI_hist_x, cum_ISI_hist_y=cum_ISI_hist_y,
                       ISIs_cells=ISIs_cells, max_ISI=max_ISI, auto_corr_cells=auto_corr_cells, max_lag=max_lag,
                       bin_width=bin_width)
    plot_for_cell_group_grid(cell_ids, cell_type_dict, plot_ISI_hist, plot_kwargs, figsize=(11, 6.5),
                            xlabel='', ylabel='', n_subplots=3, n_rows_n_columns=(1, len(cell_ids)),
                            save_dir_img=os.path.join(save_dir_img, 'ISI_hist' + str(bin_width) + '.png'))

    pl.show()