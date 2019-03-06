from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
import matplotlib.gridspec as gridspec
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, load_data, get_celltype_dict, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from analyze_in_vivo.analyze_domnisoru.autocorr.spike_time_autocorrelation import get_autocorrelation, change_bin_size_of_spike_train
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/DAP-Project/thesis/figures'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'DAP_cells'
    cell_ids = get_cell_ids_DAP_cells()
    cell_id_counter_example = 's115_0030'
    cell_ids.append(cell_id_counter_example)
    param_list = ['Vm_ljpc', 'spiketimes']
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = True
    max_ISI = 200
    max_lag = 50
    burst_ISI = 8  # ms

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

        auto_corr = get_autocorrelation(spike_train_new, max_lag_idx)
        auto_corr[max_lag_idx] = 0  # for better plotting
        auto_corr /= (np.sum(auto_corr) * bin_width)  # normalize
        auto_corr_cells[cell_idx] = auto_corr
       

    # plot
    fig = pl.figure(figsize=(10, 8))
    n_rows, n_columns = 2, 5
    outer = gridspec.GridSpec(n_rows, n_columns, height_ratios=[0.8, 0.2])

    # ISI hist., return map and auto-correlation for DAP cells
    for cell_idx in range(len(cell_ids)-1):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, cell_idx])

        ax1 = pl.subplot(inner[0])
        ax1.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
               bins[1] - bins[0], color='0.5', align='edge')
        cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[cell_idx], len(cum_ISI_hist_x[cell_idx]), max_ISI)
        cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[cell_idx], len(cum_ISI_hist_y[cell_idx]), 1.0)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
        ax1_twin.set_xlim(0, max_ISI)
        ax1_twin.set_ylim(0, 1)
        ax1_twin.set_yticks([0, 1])
        ax1.spines['right'].set_visible(True)
        ax1.set_ylabel('Rel. frequency')
        ax1.set_xlabel('ISI (ms)')
        
        ax2 = pl.subplot(inner[1])
        ax2.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
                marker='o', linestyle='', markersize=1, alpha=0.5)
        ax2.set_xlim(0, max_ISI)
        ax2.set_ylim(0, max_ISI)
        ax2.set_aspect('equal', adjustable='box-forced')
        ax2.set_ylabel('ISI[n+1] (ms)')
        ax2.set_xlabel('ISI[n] (ms)')

        ax3 = pl.subplot(inner[2])
        max3_lag_idx = to_idx(max_lag, bin_width)
        t_auto_corr = np.concatenate((np.arange(-max3_lag_idx, 0, 1), np.arange(0, max3_lag_idx + 1, 1))) * bin_width
        ax3.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_width, color='0.5', align='center')
        ax3.set_xlim(-max_lag, max_lag)
        ax3.set_ylabel('Spike-time \nautocorrelation')
        ax3.set_xlabel('Lag (ms)')

    # correlation DAP-time and peak ISI-hist
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, :2], width_ratios=[0.95, 0.05])
    ax = pl.subplot(inner[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax = pl.subplot(inner[0])
    #ax = pl.subplot(outer[1, :2], projection='3d')
    cell_idx = len(cell_ids)-1
    grid_cells = load_cell_ids(save_dir, 'grid_cells')
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells()
    DAP_time = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_time.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'grid_cells', 'peak_ISI_hist.npy'))
    peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak

    plot_with_markers(ax, DAP_time, peak_ISI_hist, grid_cells, cell_type_dict,
                      theta_cells=theta_cells, DAP_cells=DAP_cells)
    ax.plot(np.arange(0, 10), np.arange(0, 10), '0.5', linestyle='--')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_aspect('equal', adjustable='box-forced')
    ax.set_ylabel('Peak of ISI hist. (ms)')
    ax.set_xlabel('DAP time (ms)')
    l = ax.get_legend()
    l.set_bbox_to_anchor((1.0, 1.0))

    # ISI hist., return map and auto-correlation for non-bursty cell
    ax1 = pl.subplot(outer[1, 2])
    ax1.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
            bins[1] - bins[0], color='0.5', align='edge')
    cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[cell_idx], len(cum_ISI_hist_x[cell_idx]), max_ISI)
    cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[cell_idx], len(cum_ISI_hist_y[cell_idx]), 1.0)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
    ax1_twin.set_xlim(0, max_ISI)
    ax1_twin.set_ylim(0, 1)
    ax1_twin.set_yticks([0, 1])
    ax1.spines['right'].set_visible(True)
    ax1.set_ylabel('Rel. frequency')
    ax1.set_xlabel('ISI (ms)')

    ax2 = pl.subplot(outer[1, 3])
    ax2.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
             marker='o', linestyle='', markersize=1, alpha=0.5)
    ax2.set_xlim(0, max_ISI)
    ax2.set_ylim(0, max_ISI)
    ax2.set_aspect('equal', adjustable='box-forced')
    ax2.set_ylabel('ISI[n+1] (ms)')
    ax2.set_xlabel('ISI[n] (ms)')

    ax3 = pl.subplot(outer[1, 4])
    max3_lag_idx = to_idx(max_lag, bin_width)
    t_auto_corr = np.concatenate((np.arange(-max3_lag_idx, 0, 1), np.arange(0, max3_lag_idx + 1, 1))) * bin_width
    ax3.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_width, color='0.5', align='center')
    ax3.set_xlim(-max_lag, max_lag)
    ax3.set_ylabel('Spike-time \nautocorrelation')
    ax3.set_xlabel('Lag (ms)')

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_cells_bursty.png'))

    pl.show()