from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from grid_cell_stimuli import get_AP_max_idxs
from cell_characteristics import to_idx
import warnings
pl.style.use('paper')


def cross_correlate(x, y, max_lag=0):
    assert len(x) == len(y)
    cross_corr = np.zeros(2 * max_lag + 1)
    for lag in range(max_lag, 0, -1):
        cross_corr[max_lag - lag] = np.correlate(x[:-lag], y[lag:], mode='valid')[0]
    for lag in range(1, max_lag + 1, 1):
        cross_corr[max_lag + lag] = np.correlate(x[lag:], y[:-lag], mode='valid')[0]
        cross_corr[max_lag] = np.correlate(x, y, mode='valid')[0]

    assert np.all(cross_corr[:max_lag] == cross_corr[max_lag + 1:][::-1])
    return cross_corr


def auto_correlate(x, max_lag=0):
    auto_corr_lag = np.zeros(max_lag)
    for lag in range(1, max_lag, 1):
        auto_corr_lag[lag-1] = np.correlate(x[:-lag], x[lag:], mode='valid')[0]
    auto_corr_no_lag = np.array([np.correlate(x, x, mode='valid')[0]])
    auto_corr = np.concatenate((np.flipud(auto_corr_lag), auto_corr_no_lag, auto_corr_lag))
    return auto_corr


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    bin_size = 1.0  #0.5  # ms
    max_lag = 12
    use_AP_max_idxs_domnisoru = True
    in_field = False
    out_field = False
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, cell_type)
    max_lag_idx = to_idx(max_lag, bin_size)
    t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_size

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    auto_corr_cells = []
    for i, cell_id in enumerate(cell_ids):
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
        if in_field:
            in_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[in_field_len_orig[AP_max_idxs]]
        elif out_field:
            out_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'out_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[out_field_len_orig[AP_max_idxs]]
        else:
            AP_max_idxs_selected = AP_max_idxs

        # get spike-trains
        spike_train = np.zeros(len(v))
        spike_train[AP_max_idxs_selected] = 1   # for norm to firing rate:  / len(AP_max_idxs_selected)

        # change to bin size
        bin_change = bin_size / dt
        spike_train_new = np.zeros(int(round(len(spike_train) / bin_change)))
        for i in range(len(spike_train_new)):
            if sum(spike_train[i*int(bin_change):(i+1)*int(bin_change)] == 1) == 1:
                spike_train_new[i] = 1
            elif sum(spike_train[i * int(bin_change):(i + 1) * int(bin_change)] == 1) == 0:
                spike_train_new[i] = 0
            else:
                warnings.warn('More than one spike in bin!')
        #spike_train_new /= np.sum(spike_train_new)

        # pl.figure()
        # pl.plot(t[spike_train==1], spike_train[spike_train==1], 'ok')
        # pl.plot((np.arange(len(spike_train_new)) * bin_size)[spike_train_new==1], spike_train_new[spike_train_new==1], 'ob')
        # pl.show()

        # spike-time autocorrelation
        auto_corr = auto_correlate(spike_train_new, max_lag_idx)
        auto_corr /= np.sum(auto_corr)  # normalize
        auto_corr[max_lag_idx] = 0  # for better plotting
        auto_corr_cells.append(auto_corr)

        auto_corr_lags = np.zeros(max_lag_idx)
        for lag in range(1, max_lag_idx + 1):
            auto_corr_lags[lag-1] = np.correlate(spike_train_new[:-lag], spike_train_new[lag:], mode='valid')[0]

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
        pl.close('all')
        pl.figure()
        pl.bar(t_auto_corr, auto_corr, bin_size, color='0.5', align='center')
        pl.xlabel('Time (ms)')
        pl.ylabel('Spike-time autocorrelation')
        pl.xlim(-max_lag, max_lag)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'auto_corr_'+str(max_lag)+'.png'))
        #pl.show()

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
                    axes[i1, i2].bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5',
                                     align='center')
                    axes[i1, i2].set_xlim(-max_lag, max_lag)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('Time (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Spike-time \nautocorrelation')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'auto_corr_'+str(max_lag)+'.png'))
        pl.show()

    else:
        n_rows = 1 if len(cell_ids) <= 3 else 2
        n_columns = int(round(len(cell_ids)/n_rows))
        fig_height = 4.5 if len(cell_ids) <= 3 else 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, fig_height))
        if n_rows == 1:
            axes = np.array([axes])
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                    axes[i1, i2].bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5',
                                     align='center')
                    axes[i1, i2].set_xlim(-max_lag, max_lag)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('Time (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Spike-time autocorrelation')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        adjust_bottom = 0.12 if len(cell_ids) <= 3 else 0.07
        pl.subplots_adjust(left=0.07, bottom=adjust_bottom, top=0.93)
        pl.savefig(os.path.join(save_dir_img, 'auto_corr_'+str(max_lag)+'.png'))
        pl.show()